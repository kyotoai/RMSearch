# tag_pipeline.py
from __future__ import annotations
import os, json, random, math, re, argparse, sys
from typing import List, Tuple, Dict, Any, Optional

import torch
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings


# ----------------------------- #
# Helpers
# ----------------------------- #
def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (vLLM typically needs a GPU). No GPU detected.")

def _batched(lst, bs: int):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

def _json_list_from_text(text: str) -> List[str]:
    """
    Try to parse a JSON list of strings from LLM output.
    Fallbacks try to extract reasonable tags.
    """
    text = text.strip()
    # direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return [x.strip() for x in obj if x.strip()]
    except Exception:
        pass

    # fenced JSON
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return [x.strip() for x in obj if x.strip()]
        except Exception:
            pass

    # quoted items
    items = re.findall(r'"([^"]{1,80})"', text)
    if items:
        return [x.strip() for x in items if x.strip()]

    # bullets fallback
    lines = [re.sub(r"^[-*•]\s*", "", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) <= 80]
    if lines:
        return lines[:5]

    # last fallback: one tag
    return [text.splitlines()[0][:50]] if text else []


def _build_llm(model_name: str, tensor_parallel_size: int = 1, gpu_mem_util: float = 0.90) -> LLM:
    _ensure_cuda()
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_mem_util,
    )
    return llm


def _completion_json_list(llm: LLM, prompts: List[str], max_tags_per_key=6) -> List[List[str]]:
    sp = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=128,
        stop=None
    )
    outputs = llm.generate(prompts, sp)
    all_tags: List[List[str]] = []
    for out in outputs:
        text = out.outputs[0].text if out.outputs else ""
        tags = _json_list_from_text(text)
        # trim & dedup & cap
        seen = set()
        clean = []
        for t in tags:
            t = re.sub(r"[^\w\-&/ +]", "", t).strip()
            t = re.sub(r"\s+", " ", t)
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            clean.append(t)
            if len(clean) >= max_tags_per_key:
                break
        if not clean:
            clean = ["general"]
        all_tags.append(clean)
    return all_tags


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


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


# ----------------------------- #
# 1) Generate tags with vLLM
# ----------------------------- #
def generate_tag(keys: List[str], model_name: str) -> List[Dict[str, Any]]:
    """
    Returns: [{"key": str, "key_id": int, "tags": List[str]}, ...]
    """
    if not keys:
        return []

    llm = _build_llm(model_name)

    def prompt_for(k: str) -> str:
        return (
            "You are a tagging assistant.\n"
            "Task: Create 3–6 short, specific tags (1–3 words each) that describe the following key/phrase.\n"
            "Output ONLY a JSON array of strings. No commentary.\n\n"
            f"Key: \"{k}\"\n\n"
            'Example Output: ["LLM Inference", "Vector Search", "RAG"]'
        )

    prompts = [prompt_for(k) for k in keys]
    tag_lists = []
    for batch in _batched(prompts, bs=64):
        tag_lists.extend(_completion_json_list(llm, batch))

    tag_records = []
    for i, (k, tags) in enumerate(zip(keys, tag_lists)):
        tag_records.append({
            "key": k,
            "key_id": i,
            "tags": tags,
        })
    return tag_records


# ----------------------------- #
# 2) Embed tags (-> per-key embedding)
# ----------------------------- #
@torch.no_grad()
def embed_tags(
    tag_records: List[Dict[str, Any]],
    save_path: Optional[str],
    embed_model_name: str,
    quantize_precision: Optional[str] = None,   # e.g., "ubinary"
    reduce_to_dim: Optional[int] = None,        # e.g., 512
) -> torch.Tensor:
    """
    Computes an embedding per key by averaging its tag embeddings.
    Returns: embeddings: torch.FloatTensor [n_keys, embed_dim]
    """
    if not tag_records:
        emb = torch.empty(0, 0)
        if save_path:
            _save_tensor(emb, save_path)
        return emb

    device = _device()
    model = SentenceTransformer(embed_model_name).to(device)

    flat_tags: List[str] = []
    tag_meta: List[Tuple[int, int]] = []
    for rec in tag_records:
        for j, t in enumerate(rec["tags"]):
            flat_tags.append(t)
            tag_meta.append((rec["key_id"], j))

    tag_emb = model.encode(flat_tags, convert_to_tensor=True, device=device, normalize_embeddings=True)

    if quantize_precision:
        q = quantize_embeddings(tag_emb, precision=quantize_precision)
        tag_emb = torch.tensor(q, device=device)

    if reduce_to_dim is not None and 0 < reduce_to_dim < tag_emb.shape[-1]:
        mu = tag_emb.mean(dim=0, keepdim=True)
        X = tag_emb - mu
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        W = Vh[:reduce_to_dim, :].T
        tag_emb = (X @ W)
        tag_emb = _normalize(tag_emb)

    dim = tag_emb.shape[-1]
    n_keys = len(tag_records)
    key_emb = torch.zeros(n_keys, dim, device=tag_emb.device, dtype=tag_emb.dtype)
    counts = torch.zeros(n_keys, device=tag_emb.device)
    for idx, (kid, _tagid) in enumerate(tag_meta):
        key_emb[kid] += tag_emb[idx]
        counts[kid] += 1
    counts = counts.clamp_min(1.0).unsqueeze(-1)
    key_emb = _normalize(key_emb / counts).contiguous()

    if save_path:
        _save_tensor(key_emb, save_path)
    return key_emb


# ----------------------------- #
# 3) Group keys by embeddings (Torch k-means)
#     **Modified Return Spec**
# ----------------------------- #
@torch.no_grad()
def get_tag_group(
    tag_records: List[Dict[str, Any]],
    embeddings: torch.Tensor,
    n_group: int,
    n_init: int = 1,
    max_iters: int = 50,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], torch.Tensor, List[Dict[str, Any]]]:
    """
    Returns:
      tag_records: updated with "group_ids": List[int]
      centroids:   torch.FloatTensor [n_group, embed_dim]
      group_records: [
        {
          "group_id": int,
          "tags": List[str],               # deduped union of member tags
          "tag_ids": List[Tuple[int,int]]  # (key_id, tag_index)
        }, ...
      ]
    """
    if embeddings.numel() == 0 or n_group <= 0:
        for rec in tag_records:
            rec["group_ids"] = []
        return tag_records, torch.empty(0, 0), []

    torch.manual_seed(seed)
    X = embeddings.detach()
    N, D = X.shape
    n_group = max(1, min(n_group, N))

    def kmeans_run() -> Tuple[torch.Tensor, torch.Tensor]:
        # k-means++ init
        centroids = torch.empty(n_group, D, device=X.device, dtype=X.dtype)
        idx0 = torch.randint(0, N, (1,), device=X.device)
        centroids[0] = X[idx0]
        closest_dist_sq = torch.cdist(X, centroids[:1]).squeeze(-1) ** 2
        for c in range(1, n_group):
            probs = (closest_dist_sq / closest_dist_sq.sum()).clamp_min(1e-12)
            choice = torch.multinomial(probs, 1)
            centroids[c] = X[choice]
            dist_sq = torch.cdist(X, centroids[c:c+1]).squeeze(-1) ** 2
            closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq)

        prev_assign = torch.full((N,), -1, device=X.device, dtype=torch.long)
        for _ in range(max_iters):
            d = torch.cdist(X, centroids)
            assign = torch.argmin(d, dim=1)
            if torch.equal(assign, prev_assign):
                break
            prev_assign = assign
            for k in range(n_group):
                mask = (assign == k)
                if mask.any():
                    centroids[k] = X[mask].mean(dim=0)
        return assign, centroids

    best_inertia = float("inf")
    best_assign = None
    best_centroids = None
    for _ in range(n_init):
        assign, cents = kmeans_run()
        inertia = (X - cents[assign]).pow(2).sum().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_assign = assign
            best_centroids = cents

    assign = best_assign
    centroids = _normalize(best_centroids)

    # Update tag_records
    for rec in tag_records:
        gid = int(assign[rec["key_id"]].item())
        rec["group_ids"] = [gid]

    # group_records without centroids in structure (per your spec)
    groups: List[Dict[str, Any]] = []
    for gid in range(n_group):
        groups.append({
            "group_id": gid,
            "tags": [],
            "tag_ids": [],
        })

    seen_per_group = [set() for _ in range(n_group)]
    for rec in tag_records:
        gid = rec["group_ids"][0]
        for j, t in enumerate(rec["tags"]):
            low = t.lower().strip()
            if low not in seen_per_group[gid]:
                seen_per_group[gid].add(low)
                groups[gid]["tags"].append(t)
                groups[gid]["tag_ids"].append((rec["key_id"], j))

    return tag_records, centroids, groups


# ----------------------------- #
# 4) Representative tag per group (vLLM)
# ----------------------------- #
def generate_representative_tag(
    tag_records: List[Dict[str, Any]],
    group_records: List[Dict[str, Any]],
    embeddings: torch.Tensor,   # kept in signature for compatibility
    n_tag_sample: int,
    model_name: str
) -> List[Dict[str, Any]]:
    """
    Adds "representative_tag" to each group record.
    """
    if not group_records:
        return group_records

    llm = _build_llm(model_name)
    prompts = []
    for g in group_records:
        tags = g.get("tags", [])
        if not tags:
            sample = ["general"]
        else:
            sample = tags if len(tags) <= n_tag_sample else random.sample(tags, n_tag_sample)

        lines = "\n".join(f"- {t}" for t in sample)
        prompt = (
            "You are a taxonomy expert. Given the following sample tags from one cluster, "
            "produce ONE concise representative tag (≤ 6 words) that best describes them all.\n"
            "Do NOT include punctuation at the end. Output ONLY the tag text, nothing else.\n\n"
            f"Sample tags:\n{lines}\n\n"
            "Representative tag:"
        )
        prompts.append(prompt)

    sp = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=16)
    outs = llm.generate(prompts, sp)
    for g, out in zip(group_records, outs):
        txt = out.outputs[0].text.strip() if out.outputs else "General"
        txt = re.sub(r"[\n\r\"'`]", " ", txt).strip()
        txt = re.sub(r"\s+", " ", txt)
        g["representative_tag"] = txt[:60] if txt else "General"
    return group_records


# ----------------------------- #
# CLI Utilities
# ----------------------------- #
def _read_keys_from_args(keys_arg: Optional[str], keys_file: Optional[str]) -> List[str]:
    keys: List[str] = []
    if keys_file:
        with open(keys_file, "r", encoding="utf-8") as f:
            keys = [ln.strip() for ln in f if ln.strip()]
    if keys_arg:
        parts = [x.strip() for x in keys_arg.split(";")]
        keys.extend([p for p in parts if p])
    if not keys:
        raise ValueError("No keys provided. Use --keys-file or --keys 'a;b;c'.")
    return keys


# ----------------------------- #
# Debug / Example usage via CLI
# ----------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Tagging & grouping pipeline (vLLM + SentenceTransformers)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1) generate-tags
    p1 = sub.add_parser("generate-tags", help="Generate tags for keys with vLLM")
    p1.add_argument("--model", required=True, help="vLLM generation model name")
    p1.add_argument("--keys-file", help="Path to newline-separated keys")
    p1.add_argument("--keys", help="Semicolon-separated keys, e.g., 'k1;k2;k3'")
    p1.add_argument("--out-tag-recs", required=True, help="Path to save tag_records JSON")

    # 2) embed-tags
    p2 = sub.add_parser("embed-tags", help="Embed tags into per-key embeddings")
    p2.add_argument("--embed-model", required=True, help="SentenceTransformer model")
    p2.add_argument("--tag-recs", required=True, help="Input tag_records JSON")
    p2.add_argument("--embeddings-out", required=True, help="Output .pt for embeddings")
    p2.add_argument("--quantize", choices=["ubinary", "int8", "fp16"], help="Optional quantization via quantize_embeddings")
    p2.add_argument("--reduce-dim", type=int, help="Optional PCA-lite target dim (e.g., 512)")

    # 3) group-tags
    p3 = sub.add_parser("group-tags", help="Cluster keys into groups (k-means)")
    p3.add_argument("--tag-recs", required=True, help="Input tag_records JSON")
    p3.add_argument("--embeddings", required=True, help="Input embeddings .pt")
    p3.add_argument("--n-group", type=int, required=True, help="Number of groups (k)")
    p3.add_argument("--centroids-out", required=True, help="Output centroids .pt")
    p3.add_argument("--group-recs-out", required=True, help="Output group_records JSON")
    p3.add_argument("--tag-recs-out", required=True, help="Output updated tag_records JSON")
    p3.add_argument("--n-init", type=int, default=1)
    p3.add_argument("--max-iters", type=int, default=50)
    p3.add_argument("--seed", type=int, default=42)

    # 4) representative-tags
    p4 = sub.add_parser("representative-tags", help="Generate one representative tag per group")
    p4.add_argument("--model", required=True, help="vLLM generation model name")
    p4.add_argument("--group-recs", required=True, help="Input group_records JSON")
    p4.add_argument("--tag-recs", required=True, help="Input tag_records JSON (for compatibility)")
    p4.add_argument("--embeddings", required=True, help="Input embeddings .pt (kept for signature compatibility)")
    p4.add_argument("--n-tag-sample", type=int, default=6, help="Sample size of tags per group for prompting")
    p4.add_argument("--group-recs-out", required=True, help="Output updated group_records JSON")

    # 5) demo (end-to-end quick run with sample keys)
    p5 = sub.add_parser("demo", help="Run a quick demo end-to-end with sample keys")
    p5.add_argument("--model", default="Qwen2.5-3B-Instruct")
    p5.add_argument("--embed-model", default="intfloat/e5-mistral-7b-instruct")
    p5.add_argument("--save-dir", default="./artifacts")
    p5.add_argument("--n-group", type=int, default=4)
    p5.add_argument("--n-tag-sample", type=int, default=6)
    p5.add_argument("--reduce-dim", type=int, default=512)

    args = parser.parse_args()

    if args.cmd == "generate-tags":
        keys = _read_keys_from_args(args.keys, args.keys_file)
        tag_recs = generate_tag(keys, model_name=args.model)
        _save_json(tag_recs, args.out_tag_recs)
        print(f"Saved tag_records -> {args.out_tag_recs}")

    elif args.cmd == "embed-tags":
        tag_recs = _load_json(args.tag_recs)
        emb = embed_tags(
            tag_records=tag_recs,
            save_path=args.embeddings_out,
            embed_model_name=args.embed_model,
            quantize_precision=args.quantize,
            reduce_to_dim=args.reduce_dim
        )
        print(f"Embeddings shape: {tuple(emb.shape)}  saved -> {args.embeddings_out}")

    elif args.cmd == "group-tags":
        tag_recs = _load_json(args.tag_recs)
        emb = _load_tensor(args.embeddings)
        tag_recs, centroids, group_recs = get_tag_group(
            tag_records=tag_recs,
            embeddings=emb,
            n_group=args.n_group,
            n_init=args.n_init,
            max_iters=args.max_iters,
            seed=args.seed
        )
        _save_tensor(centroids, args.centroids_out)
        _save_json(group_recs, args.group_recs_out)
        _save_json(tag_recs, args.tag_recs_out)
        print(f"Centroids: {tuple(centroids.shape)} -> {args.centroids_out}")
        print(f"Saved group_records -> {args.group_recs_out}")
        print(f"Saved updated tag_records -> {args.tag_recs_out}")

    elif args.cmd == "representative-tags":
        group_recs = _load_json(args.group_recs)
        tag_recs = _load_json(args.tag_recs)  # not used directly but kept for compatibility
        emb = _load_tensor(args.embeddings)   # not used directly but kept for signature compatibility
        group_recs = generate_representative_tag(
            tag_records=tag_recs,
            group_records=group_recs,
            embeddings=emb,
            n_tag_sample=args.n_tag_sample,
            model_name=args.model
        )
        _save_json(group_recs, args.group_recs_out)
        print(f"Saved representative group_records -> {args.group_recs_out}")

    elif args.cmd == "demo":
        os.makedirs(args.save_dir, exist_ok=True)
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
        tag_path   = os.path.join(args.save_dir, "tag_recs.json")
        emb_path   = os.path.join(args.save_dir, "embeddings.pt")
        cents_path = os.path.join(args.save_dir, "centroids.pt")
        groups_path= os.path.join(args.save_dir, "group_recs.json")
        tag_upd    = os.path.join(args.save_dir, "tag_recs_grouped.json")
        groups_rep = os.path.join(args.save_dir, "group_recs_representative.json")

        print(">>> Generating tags ...")
        tag_recs = generate_tag(keys, model_name=args.model)
        _save_json(tag_recs, tag_path)
        print("tag_records ->", tag_path)

        print(">>> Embedding tags ...")
        emb = embed_tags(tag_recs, save_path=emb_path, embed_model_name=args.embed_model, reduce_to_dim=args.reduce_dim)
        print("embeddings:", tuple(emb.shape), "->", emb_path)

        print(">>> Grouping ...")
        tag_recs, cents, group_recs = get_tag_group(tag_recs, emb, n_group=args.n_group)
        _save_tensor(cents, cents_path)
        _save_json(group_recs, groups_path)
        _save_json(tag_recs, tag_upd)
        print("centroids:", tuple(cents.shape), "->", cents_path)
        print("group_records ->", groups_path)
        print("updated tag_records ->", tag_upd)

        print(">>> Representative tags ...")
        group_recs = generate_representative_tag(tag_recs, group_recs, emb, n_tag_sample=args.n_tag_sample, model_name=args.model)
        _save_json(group_recs, groups_rep)
        print("representative group_records ->", groups_rep)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
