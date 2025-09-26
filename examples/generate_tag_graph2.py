from __future__ import annotations

import argparse
import json
import os
import random
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from vllm import SamplingParams

from vllm_generate5 import LLMWorkerModel, build_llm as _build_llm_pool
from vllm_generate5 import build_llm
from vllm_embed import EmbeddingWorkerModel, build_embedding_model as _build_embed_pool, embed as _embed_with_model
from sentence_transformers.quantization import quantize_embeddings


# ----------------------------- #
# Helpers
# ----------------------------- #
def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (vLLM typically needs a GPU). No GPU detected.")


def _ensure_spawn():
    import multiprocessing as mp

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _json_list_from_text(text: str) -> List[str]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return [x.strip() for x in obj if x.strip()]
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return [x.strip() for x in obj if x.strip()]
        except Exception:
            pass

    quoted = re.findall(r'"([^\"]{1,80})"', text)
    if quoted:
        return [x.strip() for x in quoted if x.strip()]

    lines = [re.sub(r"^[-*•]\s*", "", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) <= 80]
    if lines:
        return lines[:5]
    return [text.splitlines()[0][:50]] if text else []


def _save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_tensor(t: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(t, path)


def _load_tensor(path: str) -> torch.Tensor:
    return torch.load(path, map_location=_device())


def _parse_device_groups(spec: Optional[str], tensor_parallel_size: int, num_instances: int) -> Optional[List[List[int]]]:
    if not spec:
        return None
    groups: List[List[int]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        group = [int(token) for token in chunk.split(",") if token.strip()]
        if not group:
            continue
        groups.append(group)

    if not groups:
        return None
    if len(groups) != num_instances:
        raise ValueError(
            f"device_groups mismatch: expected {num_instances} groups, got {len(groups)}"
        )
    for group in groups:
        if len(group) != tensor_parallel_size:
            raise ValueError(
                "Each device group must contain exactly "
                f"{tensor_parallel_size} device ids (got {group})"
            )
    return groups


def _json_dict_arg(text: str) -> Dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return value


def _extract_pool_settings(args: argparse.Namespace, prefix: Optional[str] = None) -> Tuple[Dict[str, Any], int, Optional[float]]:
    label = f"{prefix}_" if prefix else ""
    tps = getattr(args, f"{label}tensor_parallel_size")
    num_inst = getattr(args, f"{label}num_instances")
    device_spec = getattr(args, f"{label}device_groups")
    device_groups = _parse_device_groups(device_spec, tps, num_inst) if device_spec else None
    llm_kwargs = getattr(args, f"{label}llm_kwargs") or {}
    batch_size = getattr(args, f"{label}worker_batch_size")
    timeout = getattr(args, f"{label}timeout")
    return (
        {
            "tensor_parallel_size": tps,
            "num_instances": num_inst,
            "device_groups": device_groups,
            "llm_kwargs": llm_kwargs,
        },
        batch_size,
        timeout,
    )


@contextmanager
def _llm_pool_context(model_name: str, *, tensor_parallel_size: int, num_instances: int, device_groups: Optional[List[List[int]]] = None, llm_kwargs: Optional[Dict[str, Any]] = None) -> Iterable[LLMWorkerModel]:
    _ensure_cuda()
    _ensure_spawn()
    pool = _build_llm_pool(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        device_groups=device_groups,
        **(llm_kwargs or {}),
    )
    try:
        yield pool
    finally:
        pool.close()


@contextmanager
def _embed_pool_context(model_name: str, *, tensor_parallel_size: int, num_instances: int, device_groups: Optional[List[List[int]]] = None, llm_kwargs: Optional[Dict[str, Any]] = None) -> Iterable[EmbeddingWorkerModel]:
    _ensure_cuda()
    _ensure_spawn()
    pool = _build_embed_pool(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        device_groups=device_groups,
        **(llm_kwargs or {}),
    )
    try:
        yield pool
    finally:
        pool.close()


def _ensure_sampling(params: Optional[SamplingParams], *, temperature: float, top_p: float, max_tokens: int) -> SamplingParams:
    if params is not None:
        return params
    return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)


# ----------------------------- #
# 1) Generate tags with vLLM worker pool
# ----------------------------- #
def generate_tag(
    keys: List[str],
    model_name: str,
    *,
    pool: Optional[LLMWorkerModel] = None,
    pool_settings: Optional[Dict[str, Any]] = None,
    sampling_params: Optional[SamplingParams] = None,
    worker_batch_size: int = 8,
    timeout_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if not keys:
        return []
    prompts = [
        (
            "You are a tagging assistant.\n"
            "Task: Create 3–6 short, specific tags (1–3 words each) that describe the following key/phrase.\n"
            "Output ONLY a JSON array of strings. No commentary.\n\n"
            f"Key: \"{k}\"\n\n"
            'Example Output: ["LLM Inference", "Vector Search", "RAG"]'
        )
        for k in keys
    ]

    sampling = _ensure_sampling(sampling_params, temperature=0.3, top_p=0.9, max_tokens=128)

    owns_pool = False
    if pool is None:
        if pool_settings is None:
            raise ValueError("Either an LLM pool or pool_settings must be provided.")
        pool = _make_llm_pool_from_settings(model_name, pool_settings)
        owns_pool = True

    try:
        outputs = pool.generate(
            prompts,
            sampling_params=sampling,
            batch_size=worker_batch_size,
            timeout_s=timeout_s,
        )
    finally:
        if owns_pool and pool is not None:
            pool.close()

    all_tags: List[List[str]] = []
    for text in outputs:
        tags = _json_list_from_text(text)
        seen, clean = set(), []
        for tag in tags:
            tag = re.sub(r"[^\w\-&/ +]", "", tag).strip()
            tag = re.sub(r"\s+", " ", tag)
            if not tag or tag.lower() in seen:
                continue
            seen.add(tag.lower())
            clean.append(tag)
            if len(clean) >= 6:
                break
        if not clean:
            clean = ["general"]
        all_tags.append(clean)

    return [{"key": k, "key_id": idx, "tags": tags} for idx, (k, tags) in enumerate(zip(keys, all_tags))]


def _make_llm_pool_from_settings(model_name: str, settings: Dict[str, Any]) -> LLMWorkerModel:
    llm_kwargs = dict(settings.get("llm_kwargs") or {})
    _ensure_cuda()
    _ensure_spawn()
    return _build_llm_pool(
        model=model_name,
        tensor_parallel_size=settings["tensor_parallel_size"],
        num_instances=settings["num_instances"],
        device_groups=settings.get("device_groups"),
        **llm_kwargs,
    )


# ----------------------------- #
# 2) Embed tags using vLLM embedding worker pool
# ----------------------------- #
@torch.no_grad()
def embed_tags(
    tag_records: List[Dict[str, Any]],
    embed_model_name: str,
    *,
    pool: Optional[EmbeddingWorkerModel] = None,
    pool_settings: Optional[Dict[str, Any]] = None,
    save_path_tags: Optional[str] = None,
    save_path_tagmeta: Optional[str] = None,
    quantize_precision: Optional[str] = None,
    reduce_to_dim: Optional[int] = None,
    worker_batch_size: int = 32,
    timeout_s: Optional[float] = None,
) -> torch.Tensor:

    print("1")
    
    if not tag_records:
        tag_emb = torch.empty(0, 0)
        if save_path_tags:
            _save_tensor(tag_emb, save_path_tags)
        if save_path_tagmeta:
            _save_json([], save_path_tagmeta)
        return tag_emb

    flat_tags: List[str] = []
    tag_meta: List[Tuple[int, int]] = []
    for rec in tag_records:
        for idx, tag in enumerate(rec.get("tags", [])):
            flat_tags.append(tag)
            tag_meta.append((rec["key_id"], idx))

    print("2")

    owns_pool = False
    if pool is None:
        if pool_settings is None:
            raise ValueError("Either an embedding pool or pool_settings must be provided.")
        pool = _make_embed_pool_from_settings(embed_model_name, pool_settings)
        owns_pool = True

    print("3")

    try:
        vectors = _embed_with_model(
            pool,
            flat_tags,
            batch_size=worker_batch_size,
            timeout_s=timeout_s,
        )

        print("4")
        
    finally:
        if owns_pool and pool is not None:
            pool.close()

    device = _device()
    if vectors:
        tag_emb = torch.tensor(vectors, dtype=torch.float32, device=device)
    else:
        tag_emb = torch.empty(0, 0, dtype=torch.float32, device=device)

    if quantize_precision:
        q = quantize_embeddings(tag_emb.cpu(), precision=quantize_precision)
        tag_emb = torch.tensor(q, device=device)

    if reduce_to_dim is not None and 0 < reduce_to_dim < tag_emb.shape[-1]:
        mu = tag_emb.mean(dim=0, keepdim=True)
        X = tag_emb - mu
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        W = Vh[:reduce_to_dim, :].T
        tag_emb = _normalize(X @ W)

    if save_path_tags:
        _save_tensor(tag_emb, save_path_tags)
    if save_path_tagmeta:
        _save_json(tag_meta, save_path_tagmeta)

    return tag_emb


def _make_embed_pool_from_settings(model_name: str, settings: Dict[str, Any]) -> EmbeddingWorkerModel:
    embed_kwargs = dict(settings.get("llm_kwargs") or {})
    _ensure_cuda()
    _ensure_spawn()
    return _build_embed_pool(
        model_name=model_name,
        tensor_parallel_size=settings["tensor_parallel_size"],
        num_instances=settings["num_instances"],
        device_groups=settings.get("device_groups"),
        **embed_kwargs,
    )


# ----------------------------- #
# 3) Group tags (unchanged k-means implementation)
# ----------------------------- #
@torch.no_grad()
def get_tag_group(
    tag_records: List[Dict[str, Any]],
    embeddings: torch.Tensor,
    n_group: int,
    n_init: int = 1,
    max_iters: int = 50,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], torch.Tensor, List[Dict[str, Any]]]:
    flat_meta: List[Tuple[int, int]] = []
    flat_tags: List[str] = []
    for rec in tag_records:
        for idx, tag in enumerate(rec.get("tags", [])):
            flat_meta.append((rec["key_id"], idx))
            flat_tags.append(tag)

    total_tags = len(flat_meta)
    if total_tags == 0 or embeddings.numel() == 0:
        for rec in tag_records:
            rec["group_ids"] = [0] * len(rec.get("tags", []))
        return tag_records, torch.empty(0, 0), []

    if embeddings.shape[0] != total_tags:
        raise ValueError(
            "get_tag_group expects tag-level embeddings: "
            f"got {embeddings.shape[0]} rows but {total_tags} tags present."
        )

    torch.manual_seed(seed)
    X = embeddings.detach()
    N, D = X.shape
    n_group = max(1, min(n_group, N))

    def kmeans_run() -> Tuple[torch.Tensor, torch.Tensor]:
        centroids = torch.empty(n_group, D, device=X.device, dtype=X.dtype)
        idx0 = torch.randint(0, N, (1,), device=X.device)
        centroids[0] = X[idx0]
        closest_dist_sq = torch.cdist(X, centroids[:1]).squeeze(-1) ** 2
        for c in range(1, n_group):
            probs = (closest_dist_sq / closest_dist_sq.sum()).clamp_min(1e-12)
            choice = torch.multinomial(probs, 1)
            centroids[c] = X[choice]
            dist_sq = torch.cdist(X, centroids[c : c + 1]).squeeze(-1) ** 2
            closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq)

        prev_assign = torch.full((N,), -1, device=X.device, dtype=torch.long)
        for _ in range(max_iters):
            dist = torch.cdist(X, centroids)
            assign = torch.argmin(dist, dim=1)
            if torch.equal(assign, prev_assign):
                break
            prev_assign = assign
            for k in range(n_group):
                mask = assign == k
                if mask.any():
                    centroids[k] = X[mask].mean(dim=0)
        return assign, centroids

    best_inertia = float("inf")
    best_assign: Optional[torch.Tensor] = None
    best_centroids: Optional[torch.Tensor] = None
    for _ in range(n_init):
        assign, cents = kmeans_run()
        inertia = (X - cents[assign]).pow(2).sum().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_assign = assign
            best_centroids = cents

    assert best_assign is not None and best_centroids is not None
    assign = best_assign
    centroids = _normalize(best_centroids)

    by_key: Dict[int, List[int]] = {}
    for (key_id, tag_idx), group_id in zip(flat_meta, assign.tolist()):
        by_key.setdefault(key_id, []).append(group_id)

    for rec in tag_records:
        groups = by_key.get(rec["key_id"], [])
        rec["group_ids"] = groups if len(groups) == len(rec.get("tags", [])) else groups[: len(rec.get("tags", []))]
        if len(rec["group_ids"]) < len(rec.get("tags", [])):
            rec["group_ids"].extend([0] * (len(rec["tags"]) - len(rec["group_ids"])) )

    groups: List[Dict[str, Any]] = [{"group_id": idx, "tags": [], "tag_ids": []} for idx in range(n_group)]
    seen_per_group = [set() for _ in range(n_group)]
    for (key_id, tag_idx), group_id in zip(flat_meta, assign.tolist()):
        t = tag_records[key_id]["tags"][tag_idx]
        ''' # to avoid overlapping, but in this case tags and tag_ids should correspond each other, so it shouldn't be avoided
        low = t.lower().strip()
        if low not in seen_per_group[g]:
            seen_per_group[g].add(low)
            groups[g]["tags"].append(t)
        '''
        groups[group_id]["tags"].append(t)
        groups[group_id]["tag_ids"].append((key_id, tag_idx))

    return tag_records, centroids, groups


# ----------------------------- #
# 4) Representative tag per group using LLM worker pool
# ----------------------------- #
def generate_representative_tag(
    tag_records: List[Dict[str, Any]],
    group_records: List[Dict[str, Any]],
    *,
    model_name: str,
    #pool: Optional[LLMWorkerModel] = None,
    #pool_settings: Optional[Dict[str, Any]] = None,
    sampling_params: Optional[SamplingParams] = None,
    tensor_parallel_size: int = 1,
    num_instances: int = 1,
    worker_batch_size: int = 8,
    timeout_s: Optional[float] = None,
    n_tag_sample: int,
) -> List[Dict[str, Any]]:
    if not group_records:
        return group_records

    prompts: List[str] = []
    for record in group_records:
        tags = record.get("tags", [])
        sample = ["general"] if not tags else (tags if len(tags) <= n_tag_sample else random.sample(tags, n_tag_sample))
        lines = "\n".join(f"- {tag}" for tag in sample)
        prompts.append(
            "You are a taxonomy expert. Given the following sample tags from one cluster, "
            "produce ONE concise representative tag (≤ 6 words) that best describes them all.\n"
            "Do NOT include punctuation at the end. Output ONLY the tag text, nothing else.\n\n"
            f"Sample tags:\n{lines}\n\nRepresentative tag:"
        )

    sampling = _ensure_sampling(sampling_params, temperature=0.4, top_p=0.9, max_tokens=16)

    device_groups = []
    device_id = 0
    for dp in range(num_instances):
        device_group = []
        for tp in range(tensor_parallel_size):
            device_group.append(device_id)
            device_id += 1
    
        device_groups.append(device_group)
    
    model = build_llm(
        model_name=model_name,
        tensor_parallel_size=len(device_groups[0]),
        num_instances=len(device_groups),
        device_groups=device_groups,
        max_model_len=2500,
        max_num_seqs=64,
        gpu_memory_utilization=0.90,
    )

    try:
        outputs = model.generate(
            prompts,
            sampling_params=sampling,
            batch_size=worker_batch_size,
            timeout_s=timeout_s,
        )
    finally:
        model.close()

    for record, text in zip(group_records, outputs):
        cleaned = re.sub(r"[\n\r\"'`]+", " ", text or "").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        record["representative_tag"] = cleaned[:60] if cleaned else "General"
    return group_records


@torch.no_grad()
def generate_tag_tree(
    group_records: List[Dict[str, Any]],
    centroids: torch.Tensor,
    tree_struc: List[int],
    n_tag_sample: int,
    *,
    model_name: str = "Qwen2.5-3B-Instruct",
    sampling_params: Optional[SamplingParams] = None,
    tensor_parallel_size: int = 1,
    num_instances: int = 1,
    worker_batch_size: int = 8,
    timeout_s: Optional[float] = None,
    n_init: int = 1,
    max_iters: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Build a hierarchical tag tree by repeatedly clustering existing centroids.
    This version avoids OOM by:
      - building the LLM once,
      - doing all representative-tag generations via a single reused model,
      - never calling generate_representative_tag.
    """
    # ---------- small helpers (local, no external changes) ----------
    def _normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def _kmeans(X: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (assignments [N], centroids [K,D]) similar to get_tag_group()."""
        torch.manual_seed(seed)
        N, D = X.shape
        K = max(1, min(K, N))

        # k-means++ init
        cents = torch.empty(K, D, device=X.device, dtype=X.dtype)
        idx0 = torch.randint(0, N, (1,), device=X.device)
        cents[0] = X[idx0]
        closest_dist_sq = torch.cdist(X, cents[:1]).squeeze(-1) ** 2
        for c in range(1, K):
            probs = (closest_dist_sq / closest_dist_sq.sum()).clamp_min(1e-12)
            choice = torch.multinomial(probs, 1)
            cents[c] = X[choice]
            dist_sq = torch.cdist(X, cents[c:c+1]).squeeze(-1) ** 2
            closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq)

        best_inertia = float("inf")
        best_assign, best_cents = None, None

        for _ in range(n_init):
            prev_assign = torch.full((N,), -1, device=X.device, dtype=torch.long)
            cur_cents = cents.clone()
            for _it in range(max_iters):
                dist = torch.cdist(X, cur_cents)
                assign = torch.argmin(dist, dim=1)
                if torch.equal(assign, prev_assign):
                    break
                prev_assign = assign
                for k in range(K):
                    mask = assign == k
                    if mask.any():
                        cur_cents[k] = X[mask].mean(dim=0)

            inertia = (X - cur_cents[assign]).pow(2).sum().item()
            if inertia < best_inertia:
                best_inertia = inertia
                best_assign = assign
                best_cents = cur_cents

        assert best_assign is not None and best_cents is not None
        return best_assign, _normalize_rows(best_cents)

    def _mk_prompts_from_tag_lists(tag_lists: List[List[str]]) -> List[str]:
        prompts: List[str] = []
        for tags in tag_lists:
            sample = tags if len(tags) <= n_tag_sample else random.sample(tags, n_tag_sample)
            if not sample:
                sample = ["general"]
            lines = "\n".join(f"- {t}" for t in sample)
            prompts.append(
                "You are a taxonomy expert. Given the following sample tags from one cluster, "
                "produce ONE concise representative tag (≤ 6 words) that best describes them all.\n"
                "Do NOT include punctuation at the end. Output ONLY the tag text, nothing else.\n\n"
                f"Sample tags:\n{lines}\n\nRepresentative tag:"
            )
        return prompts

    def _clean_rep_text(x: str) -> str:
        cleaned = re.sub(r"[\n\r\"'`]+", " ", x or "").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned[:60] if cleaned else "General"

    # ---------- input checks ----------
    if not isinstance(tree_struc, list) or not tree_struc:
        raise ValueError("tree_struc must be a non-empty list of integers.")
    if centroids is None or centroids.numel() == 0:
        return []
    if len(group_records) != centroids.shape[0]:
        raise ValueError(
            f"Length mismatch: {len(group_records)} group_records vs centroids {tuple(centroids.shape)}"
        )

    # ---------- build LLM ONCE ----------
    sampling = _ensure_sampling(sampling_params, temperature=0.4, top_p=0.9, max_tokens=16)

    device_groups = []
    device_id = 0
    for _dp in range(num_instances):
        group = []
        for _tp in range(tensor_parallel_size):
            group.append(device_id)
            device_id += 1
        device_groups.append(group)

    model = build_llm(
        model_name=model_name,
        tensor_parallel_size=len(device_groups[0]),
        num_instances=len(device_groups),
        device_groups=device_groups,
        max_model_len=2500,
        max_num_seqs=64,
        gpu_memory_utilization=0.90,
    )

    try:
        # ---------- ensure leaf representative tags (using single model instance) ----------
        need_leaf = any("representative_tag" not in rec or not rec["representative_tag"] for rec in group_records)
        if need_leaf:
            leaf_tag_lists = [rec.get("tags", []) for rec in group_records]
            leaf_prompts = _mk_prompts_from_tag_lists(leaf_tag_lists)
            leaf_out = model.generate(
                leaf_prompts,
                sampling_params=sampling,
                batch_size=worker_batch_size,
                timeout_s=timeout_s,
            )
            for rec, text in zip(group_records, leaf_out):
                rec["representative_tag"] = _clean_rep_text(text)

        # build initial leaf nodes
        cur_nodes: List[Dict[str, Any]] = [
            {"tag": rec.get("representative_tag", "General"), "children": []}
            for rec in group_records
        ]
        X = _normalize_rows(centroids.detach())

        # ---------- hierarchical levels ----------
        for level_target in tree_struc:
            if len(cur_nodes) <= level_target:
                # no clustering needed; keep nodes as-is, but still a "level" structurally
                # (no new representative tags to compute)
                continue

            # cluster to level_target
            assign, cents = _kmeans(X, int(level_target))
            K = cents.shape[0]

            # gather children per parent cluster
            clusters: List[List[int]] = [[] for _ in range(K)]
            for i, a in enumerate(assign.tolist()):
                clusters[a].append(i)

            # prepare prompts for parent representative tags (from child node tags)
            parent_tag_lists: List[List[str]] = [[cur_nodes[i]["tag"] for i in child_idxs] for child_idxs in clusters]
            parent_prompts = _mk_prompts_from_tag_lists(parent_tag_lists)

            parent_out = model.generate(
                parent_prompts,
                sampling_params=sampling,
                batch_size=worker_batch_size,
                timeout_s=timeout_s,
            )
            parent_tags = [_clean_rep_text(t) for t in parent_out]

            # assemble parent nodes
            parent_nodes: List[Dict[str, Any]] = []
            for ptag, child_idxs in zip(parent_tags, clusters):
                parent_nodes.append(
                    {
                        "tag": ptag,
                        "children": [cur_nodes[i] for i in child_idxs],
                    }
                )

            # move up one level
            cur_nodes = parent_nodes
            X = cents

        # cur_nodes is the top level
        return cur_nodes

    finally:
        model.close()


# ----------------------------- #
# CLI helpers
# ----------------------------- #
def _read_keys_from_args(keys_arg: Optional[str], keys_file: Optional[str]) -> List[str]:
    keys: List[str] = []
    if keys_file:
        with open(keys_file, "r", encoding="utf-8") as f:
            keys.extend([line.strip() for line in f if line.strip()])
    if keys_arg:
        parts = [piece.strip() for piece in keys_arg.split(";")]
        keys.extend([p for p in parts if p])
    if not keys:
        raise ValueError("No keys provided. Use --keys-file or --keys 'a;b;c'.")
    return keys


def _add_pool_args(parser: argparse.ArgumentParser, prefix: Optional[str] = None, *, default_batch_size: int = 8, description: str = "LLM") -> None:
    label = f"{prefix}-" if prefix else ""
    parser.add_argument(f"--{label}tensor-parallel-size", type=int, default=1, help=f"{description} tensor_parallel_size (per instance).")
    parser.add_argument(f"--{label}num-instances", type=int, default=1, help=f"Number of {description} worker instances.")
    parser.add_argument(
        f"--{label}device-groups",
        type=str,
        help="Semi-colon separated GPU groups, e.g. '0,1;2,3' for two workers with tensor parallel 2.",
    )
    parser.add_argument(f"--{label}worker-batch-size", type=int, default=default_batch_size, help=f"Prompts per batch dispatched to each {description} worker.")
    parser.add_argument(f"--{label}timeout", type=float, help="Timeout in seconds for a worker batch to complete.")
    parser.add_argument(
        f"--{label}llm-kwargs",
        type=_json_dict_arg,
        default=None,
        help="Additional JSON kwargs forwarded to vLLM build helper (e.g. '{\"max_model_len\": 4096}').",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tagging & grouping pipeline (vLLM worker pools)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("generate-tags", help="Generate tags for keys using vLLM worker pool")
    p1.add_argument("--model", required=True)
    p1.add_argument("--keys-file")
    p1.add_argument("--keys")
    p1.add_argument("--out-tag-recs", required=True)
    _add_pool_args(p1)

    p2 = sub.add_parser("embed-tags", help="Embed tags via vLLM embedding worker pool")
    p2.add_argument("--embed-model", required=True)
    p2.add_argument("--tag-recs", required=True)
    p2.add_argument("--embeddings-tags-out", required=True)
    p2.add_argument("--tag-meta-out", required=True)
    p2.add_argument("--quantize", choices=["ubinary", "int8", "fp16"])
    p2.add_argument("--reduce-dim", type=int)
    _add_pool_args(p2, default_batch_size=32, description="embedding")

    p3 = sub.add_parser("group-tags", help="Cluster tags into groups (k-means)")
    p3.add_argument("--tag-recs", required=True)
    p3.add_argument("--embeddings-tags", required=True)
    p3.add_argument("--n-group", type=int, required=True)
    p3.add_argument("--centroids-out", required=True)
    p3.add_argument("--group-recs-out", required=True)
    p3.add_argument("--tag-recs-out", required=True)
    p3.add_argument("--n-init", type=int, default=1)
    p3.add_argument("--max-iters", type=int, default=50)
    p3.add_argument("--seed", type=int, default=42)

    p4 = sub.add_parser("representative-tags", help="Generate representative tag per group")
    p4.add_argument("--model", required=True)
    p4.add_argument("--group-recs", required=True)
    p4.add_argument("--tag-recs", required=True)
    p4.add_argument("--n-tag-sample", type=int, default=6)
    p4.add_argument("--group-recs-out", required=True)
    _add_pool_args(p4)

    p5 = sub.add_parser("demo", help="End-to-end demo pipeline")
    p5.add_argument("--model", default="Qwen2.5-3B-Instruct")
    p5.add_argument("--embed-model", default="intfloat/e5-mistral-7b-instruct")
    p5.add_argument("--save-dir", default="./artifacts")
    p5.add_argument("--n-group", type=int, default=4)
    p5.add_argument("--n-tag-sample", type=int, default=6)
    p5.add_argument("--reduce-dim", type=int, default=512)
    _add_pool_args(p5, prefix="gen", description="generation")
    _add_pool_args(p5, prefix="embed", default_batch_size=32, description="embedding")

    args = parser.parse_args()

    if args.cmd == "generate-tags":
        pool_settings, batch_size, timeout = _extract_pool_settings(args)
        keys = _read_keys_from_args(args.keys, args.keys_file)
        with _llm_pool_context(args.model, **pool_settings) as pool:
            tag_recs = generate_tag(
                keys,
                model_name=args.model,
                pool=pool,
                worker_batch_size=batch_size,
                timeout_s=timeout,
            )
        _save_json(tag_recs, args.out_tag_recs)
        print(f"Saved tag_records -> {args.out_tag_recs}")

    elif args.cmd == "embed-tags":
        pool_settings, batch_size, timeout = _extract_pool_settings(args)
        tag_recs = _load_json(args.tag_recs)
        with _embed_pool_context(args.embed_model, **pool_settings) as pool:
            tag_emb = embed_tags(
                tag_records=tag_recs,
                embed_model_name=args.embed_model,
                pool=pool,
                worker_batch_size=batch_size,
                timeout_s=timeout,
                save_path_tags=args.embeddings_tags_out,
                save_path_tagmeta=args.tag_meta_out,
                quantize_precision=args.quantize,
                reduce_to_dim=args.reduce_dim,
            )
        print(f"Tag embeddings: {tuple(tag_emb.shape)} -> {args.embeddings_tags_out}")
        print(f"Tag meta saved -> {args.tag_meta_out}")

    elif args.cmd == "group-tags":
        tag_recs = _load_json(args.tag_recs)
        tag_emb = _load_tensor(args.embeddings_tags)
        tag_recs, centroids, group_recs = get_tag_group(
            tag_records=tag_recs,
            embeddings=tag_emb,
            n_group=args.n_group,
            n_init=args.n_init,
            max_iters=args.max_iters,
            seed=args.seed,
        )
        _save_tensor(centroids, args.centroids_out)
        _save_json(group_recs, args.group_recs_out)
        _save_json(tag_recs, args.tag_recs_out)
        print(f"Centroids: {tuple(centroids.shape)} -> {args.centroids_out}")
        print(f"Saved group_records -> {args.group_recs_out}")
        print(f"Saved updated tag_records -> {args.tag_recs_out}")

    elif args.cmd == "representative-tags":
        pool_settings, batch_size, timeout = _extract_pool_settings(args)
        group_recs = _load_json(args.group_recs)
        tag_recs = _load_json(args.tag_recs)
        with _llm_pool_context(args.model, **pool_settings) as pool:
            group_recs = generate_representative_tag(
                tag_records=tag_recs,
                group_records=group_recs,
                model_name=args.model,
                pool=pool,
                worker_batch_size=batch_size,
                timeout_s=timeout,
                n_tag_sample=args.n_tag_sample,
            )
        _save_json(group_recs, args.group_recs_out)
        print(f"Saved representative group_records -> {args.group_recs_out}")

    elif args.cmd == "demo":
        os.makedirs(args.save_dir, exist_ok=True)
        gen_settings, gen_batch, gen_timeout = _extract_pool_settings(args, prefix="gen")
        embed_settings, embed_batch, embed_timeout = _extract_pool_settings(args, prefix="embed")

        keys = [
            "Graph-based retrieval augmentation for enterprise documents",
            "Reward models for search result re-ranking",
            "Neural sparse indexing for web-scale retrieval",
            "Semantic tagging of legal contracts",
            "LLM orchestration with multi-agent planners",
            "Efficient RAG with vector + keyword hybrid",
            "Biomedical literature triage with MeSH terms",
            "GPU-efficient vLLM serving on multi-GPU",
            "Evaluation of negotiation agents with self-play",
            "Knowledge graph construction from PDFs",
        ]

        tag_path = os.path.join(args.save_dir, "tag_recs.json")
        tag_emb_out = os.path.join(args.save_dir, "embeddings_tags.pt")
        tag_meta_out = os.path.join(args.save_dir, "tag_meta.json")
        cents_path = os.path.join(args.save_dir, "centroids.pt")
        groups_path = os.path.join(args.save_dir, "group_recs.json")
        tag_upd = os.path.join(args.save_dir, "tag_recs_grouped.json")
        groups_rep = os.path.join(args.save_dir, "group_recs_representative.json")

        print(">>> Generating tags ...")
        with _llm_pool_context(args.model, **gen_settings) as pool:
            tag_recs = generate_tag(
                keys,
                model_name=args.model,
                pool=pool,
                worker_batch_size=gen_batch,
                timeout_s=gen_timeout,
            )
        _save_json(tag_recs, tag_path)

        print(">>> Embedding (batched) ...")
        with _embed_pool_context(args.embed_model, **embed_settings) as pool:
            tag_emb = embed_tags(
                tag_records=tag_recs,
                embed_model_name=args.embed_model,
                pool=pool,
                worker_batch_size=embed_batch,
                timeout_s=embed_timeout,
                save_path_tags=tag_emb_out,
                save_path_tagmeta=tag_meta_out,
                reduce_to_dim=args.reduce_dim,
            )
        print("Tag embeddings:", tuple(tag_emb.shape), "->", tag_emb_out)

        print(">>> Grouping per-tag ...")
        tag_recs, cents, group_recs = get_tag_group(tag_recs, tag_emb, n_group=args.n_group)
        _save_tensor(cents, cents_path)
        _save_json(group_recs, groups_path)
        _save_json(tag_recs, tag_upd)
        print("Centroids:", tuple(cents.shape), "->", cents_path)

        print(">>> Representative tags ...")
        with _llm_pool_context(args.model, **gen_settings) as pool:
            group_recs = generate_representative_tag(
                tag_records=tag_recs,
                group_records=group_recs,
                model_name=args.model,
                pool=pool,
                worker_batch_size=gen_batch,
                timeout_s=gen_timeout,
                n_tag_sample=args.n_tag_sample,
            )
        _save_json(group_recs, groups_rep)
        print("Representative group_records ->", groups_rep)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
