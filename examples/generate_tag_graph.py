# tag_pipeline.py
from __future__ import annotations
import os, json, random, re, argparse
from typing import List, Tuple, Dict, Any, Optional

import torch
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
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
    text = text.strip()
    # try JSON
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
    # bullets
    lines = [re.sub(r"^[-*•]\s*", "", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) <= 80]
    if lines:
        return lines[:5]
    return [text.splitlines()[0][:50]] if text else []

def _build_llm(model_name: str, tensor_parallel_size: int = 1, gpu_mem_util: float = 0.90) -> LLM:
    _ensure_cuda()
    return LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_mem_util,
    )

def _completion_json_list(llm: LLM, prompts: List[str], max_tags_per_key=6) -> List[List[str]]:
    sp = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=128)
    outputs = llm.generate(prompts, sp)
    all_tags: List[List[str]] = []
    for out in outputs:
        text = out.outputs[0].text if out.outputs else ""
        tags = _json_list_from_text(text)
        # clean + cap
        seen, clean = set(), []
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
def generate_tag(keys: List[str], model_name: str, batch_size = 1000) -> List[Dict[str, Any]]:
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
    for batch in _batched(prompts, bs=batch_size):
        tag_lists.extend(_completion_json_list(llm, batch))

    tag_records = []
    for i, (k, tags) in enumerate(zip(keys, tag_lists)):
        tag_records.append({"key": k, "key_id": i, "tags": tags})
    return tag_records


# ----------------------------- #
# 2) Embed tags (per-tag + per-key) with batching
# ----------------------------- #
@torch.no_grad()
def embed_tags(
    tag_records: List[Dict[str, Any]],
    #save_path_keys: Optional[str],
    embed_model_name: str,
    *,
    save_path_tags: Optional[str] = None,
    save_path_tagmeta: Optional[str] = None,
    quantize_precision: Optional[str] = None,   # e.g., "ubinary"
    reduce_to_dim: Optional[int] = None,        # e.g., 512
    batch_size: int = 256
) -> torch.Tensor:
    """
    - Produces **tag-level embeddings** (one per tag) with robust batching.
    - Aggregates to **key-level embeddings** by averaging each key's tags.
    Saves (optional):
      save_path_tags:  torch.Tensor [n_tags_total, dim]  (tag embeddings)
      save_path_tagmeta: JSON list of (key_id, tag_idx) of length n_tags_total
      save_path_keys: torch.Tensor [n_keys, dim] (key embeddings)
    Returns: key_embeddings (torch.Tensor [n_keys, dim])
    """
    if not tag_records:
        key_emb = torch.empty(0, 0)
        if save_path_keys: _save_tensor(key_emb, save_path_keys)
        if save_path_tags: _save_tensor(key_emb, save_path_tags)
        if save_path_tagmeta: _save_json([], save_path_tagmeta)
        return key_emb

    device = _device()
    model = SentenceTransformer(embed_model_name).to(device)

    # Flatten tags with provenance
    flat_tags: List[str] = []
    tag_meta: List[Tuple[int, int]] = []
    for rec in tag_records:
        for j, t in enumerate(rec["tags"]):
            flat_tags.append(t)
            tag_meta.append((rec["key_id"], j))

    # Encode with safe batching (and auto backoff if OOM)
    def encode_batched(texts: List[str], bs: int) -> torch.Tensor:
        current_bs = max(1, bs)
        embs: List[torch.Tensor] = []
        i = 0
        while i < len(texts):
            chunk = texts[i:i+current_bs]
            try:
                e = model.encode(
                    chunk,
                    convert_to_tensor=True,
                    device=device,
                    normalize_embeddings=True,
                    batch_size=current_bs,
                    show_progress_bar=False
                )
                embs.append(e)
                i += current_bs
            except RuntimeError as err:
                if "CUDA out of memory" in str(err) and current_bs > 1:
                    torch.cuda.empty_cache()
                    current_bs = max(1, current_bs // 2)  # back off
                    continue
                raise
        return torch.cat(embs, dim=0) if embs else torch.empty(0, model.get_sentence_embedding_dimension(), device=device)

    tag_emb = encode_batched(flat_tags, batch_size)

    # optional quantization
    if quantize_precision:
        q = quantize_embeddings(tag_emb, precision=quantize_precision)
        tag_emb = torch.tensor(q, device=device)

    # optional dim reduce (PCA-lite via SVD)
    if reduce_to_dim is not None and 0 < reduce_to_dim < tag_emb.shape[-1]:
        mu = tag_emb.mean(dim=0, keepdim=True)
        X = tag_emb - mu
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        W = Vh[:reduce_to_dim, :].T
        tag_emb = _normalize(X @ W)

    # Save tag-level things if paths provided
    if save_path_tags:
        _save_tensor(tag_emb, save_path_tags)
    if save_path_tagmeta:
        _save_json(tag_meta, save_path_tagmeta)

    return tag_emb

    '''
    # Aggregate to per-key embeddings
    dim = tag_emb.shape[-1]
    n_keys = len(tag_records)
    key_emb = torch.zeros(n_keys, dim, device=tag_emb.device, dtype=tag_emb.dtype)
    counts = torch.zeros(n_keys, device=tag_emb.device)
    for idx, (kid, _tagid) in enumerate(tag_meta):
        key_emb[kid] += tag_emb[idx]
        counts[kid] += 1
    key_emb = _normalize(key_emb / counts.clamp_min(1.0).unsqueeze(-1)).contiguous()

    if save_path_keys:
        _save_tensor(key_emb, save_path_keys)
    return key_emb
    '''


# ----------------------------- #
# 3) Group **tags** (Torch k-means)
#     Return spec: tag_records, centroids, group_records
# ----------------------------- #
@torch.no_grad()
def get_tag_group(
    tag_records: List[Dict[str, Any]],
    embeddings: torch.Tensor,   # MUST be tag-level embeddings (one per tag)
    n_group: int,
    n_init: int = 1,
    max_iters: int = 50,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], torch.Tensor, List[Dict[str, Any]]]:
    """
    Clusters INDIVIDUAL TAG EMBEDDINGS.
    Ensures len(rec["group_ids"]) == len(rec["tags"]) by assigning one group per tag.

    Returns:
      tag_records: each dict has "group_ids": List[int] aligned with its "tags"
      centroids:   torch.FloatTensor [n_group, embed_dim]
      group_records: [{"group_id":int, "tags":List[str], "tag_ids":List[Tuple[int,int]]}]
    """
    # Build meta map in the SAME order used for embeddings
    flat_meta: List[Tuple[int, int]] = []  # (key_id, tag_idx)
    flat_tags: List[str] = []
    for rec in tag_records:
        for j, t in enumerate(rec["tags"]):
            flat_meta.append((rec["key_id"], j))
            flat_tags.append(t)

    T = len(flat_meta)
    if T == 0 or embeddings.numel() == 0:
        for rec in tag_records:
            rec["group_ids"] = [0] * len(rec.get("tags", []))
        return tag_records, torch.empty(0, 0), []

    if embeddings.shape[0] != T:
        raise ValueError(
            f"get_tag_group expects tag-level embeddings: got {embeddings.shape[0]} rows, "
            f"but there are {T} tags across tag_records."
        )

    torch.manual_seed(seed)
    X = embeddings.detach()
    N, D = X.shape
    n_group = max(1, min(n_group, N))

    # k-means++ init + Lloyd iterations
    def kmeans_run() -> Tuple[torch.Tensor, torch.Tensor]:
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

    # Fill group_ids aligned to tags for each record
    by_key: Dict[int, List[int]] = {}
    for (kid, tidx), g in zip(flat_meta, assign.tolist()):
        by_key.setdefault(kid, []).append(g)
    # Keep original order per key
    for rec in tag_records:
        rec["group_ids"] = by_key.get(rec["key_id"], [0] * len(rec["tags"]))
        if len(rec["group_ids"]) != len(rec["tags"]):
            # Should not happen, but guard
            rec["group_ids"] = rec["group_ids"][:len(rec["tags"])] + [0] * max(0, len(rec["tags"]) - len(rec["group_ids"]))

    # Build group_records
    groups: List[Dict[str, Any]] = [{"group_id": gid, "tags": [], "tag_ids": []} for gid in range(n_group)]
    seen_per_group = [set() for _ in range(n_group)]
    for (kid, tidx), g in zip(flat_meta, assign.tolist()):
        t = tag_records[kid]["tags"][tidx]
        ''' # to avoid overlapping, but in this case tags and tag_ids should correspond each other, so it shouldn't be avoided
        low = t.lower().strip()
        if low not in seen_per_group[g]:
            seen_per_group[g].add(low)
            groups[g]["tags"].append(t)
        '''
        groups[g]["tags"].append(t)
        groups[g]["tag_ids"].append((kid, tidx))

    return tag_records, centroids, groups


# ----------------------------- #
# 4) Representative tag per group (vLLM)
# ----------------------------- #
def generate_representative_tag(
    tag_records: List[Dict[str, Any]],
    group_records: List[Dict[str, Any]],
    embeddings: torch.Tensor,   # kept for signature compatibility
    n_tag_sample: int,
    model_name: str
) -> List[Dict[str, Any]]:
    if not group_records:
        return group_records
    llm = _build_llm(model_name)
    prompts = []
    for g in group_records:
        tags = g.get("tags", [])
        sample = ["general"] if not tags else (tags if len(tags) <= n_tag_sample else random.sample(tags, n_tag_sample))
        lines = "\n".join(f"- {t}" for t in sample)
        prompt = (
            "You are a taxonomy expert. Given the following sample tags from one cluster, "
            "produce ONE concise representative tag (≤ 6 words) that best describes them all.\n"
            "Do NOT include punctuation at the end. Output ONLY the tag text, nothing else.\n\n"
            f"Sample tags:\n{lines}\n\nRepresentative tag:"
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
# CLI
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

def main():
    parser = argparse.ArgumentParser(description="Tagging & grouping pipeline (per-tag clustering)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1) generate-tags
    p1 = sub.add_parser("generate-tags", help="Generate tags for keys with vLLM")
    p1.add_argument("--model", required=True)
    p1.add_argument("--keys-file")
    p1.add_argument("--keys")
    p1.add_argument("--out-tag-recs", required=True)

    # 2) embed-tags (now saves tag-level emb + meta; returns key emb)
    p2 = sub.add_parser("embed-tags", help="Embed tags (tag-level) and aggregate to key-level")
    p2.add_argument("--embed-model", required=True)
    p2.add_argument("--tag-recs", required=True)
    p2.add_argument("--embeddings-keys-out", required=True, help="Output path for key-level embeddings (.pt)")
    p2.add_argument("--embeddings-tags-out", required=True, help="Output path for tag-level embeddings (.pt)")
    p2.add_argument("--tag-meta-out", required=True, help="Output path for tag meta JSON [(key_id, tag_idx), ...]")
    p2.add_argument("--quantize", choices=["ubinary", "int8", "fp16"])
    p2.add_argument("--reduce-dim", type=int)
    p2.add_argument("--batch-size", type=int, default=256)

    # 3) group-tags (expects TAG-LEVEL embeddings)
    p3 = sub.add_parser("group-tags", help="Cluster tags into groups (k-means)")
    p3.add_argument("--tag-recs", required=True)
    p3.add_argument("--embeddings-tags", required=True, help="Input tag-level embeddings (.pt)")
    p3.add_argument("--n-group", type=int, required=True)
    p3.add_argument("--centroids-out", required=True)
    p3.add_argument("--group-recs-out", required=True)
    p3.add_argument("--tag-recs-out", required=True)
    p3.add_argument("--n-init", type=int, default=1)
    p3.add_argument("--max-iters", type=int, default=50)
    p3.add_argument("--seed", type=int, default=42)

    # 4) representative-tags
    p4 = sub.add_parser("representative-tags", help="Generate a representative tag per group")
    p4.add_argument("--model", required=True)
    p4.add_argument("--group-recs", required=True)
    p4.add_argument("--tag-recs", required=True)
    p4.add_argument("--embeddings-tags", required=True)  # not used but kept for parity
    p4.add_argument("--n-tag-sample", type=int, default=6)
    p4.add_argument("--group-recs-out", required=True)

    # 5) demo
    p5 = sub.add_parser("demo", help="End-to-end demo with sample keys (per-tag clustering)")
    p5.add_argument("--model", default="Qwen2.5-3B-Instruct")
    p5.add_argument("--embed-model", default="intfloat/e5-mistral-7b-instruct")
    p5.add_argument("--save-dir", default="./artifacts")
    p5.add_argument("--n-group", type=int, default=4)
    p5.add_argument("--n-tag-sample", type=int, default=6)
    p5.add_argument("--reduce-dim", type=int, default=512)
    p5.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()

    if args.cmd == "generate-tags":
        keys = _read_keys_from_args(args.keys, args.keys_file)
        tag_recs = generate_tag(keys, model_name=args.model)
        _save_json(tag_recs, args.out_tag_recs)
        print(f"Saved tag_records -> {args.out_tag_recs}")

    elif args.cmd == "embed-tags":
        tag_recs = _load_json(args.tag_recs)
        key_emb = embed_tags(
            tag_records=tag_recs,
            save_path_keys=args.embeddings_keys_out,
            embed_model_name=args.embed_model,
            save_path_tags=args.embeddings_tags_out,
            save_path_tagmeta=args.tag_meta_out,
            quantize_precision=args.quantize,
            reduce_to_dim=args.reduce_dim,
            batch_size=args.batch_size
        )
        print(f"Key embeddings: {tuple(key_emb.shape)} -> {args.embeddings_keys_out}")
        print(f"Tag embeddings saved -> {args.embeddings_tags_out}")
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
        tag_recs = _load_json(args.tag_recs)
        _ = _load_tensor(args.embeddings_tags)  # kept for parity
        group_recs = generate_representative_tag(
            tag_records=tag_recs,
            group_records=group_recs,
            embeddings=torch.empty(0),  # unused
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
        tag_path    = os.path.join(args.save_dir, "tag_recs.json")
        key_emb_out = os.path.join(args.save_dir, "embeddings_keys.pt")
        tag_emb_out = os.path.join(args.save_dir, "embeddings_tags.pt")
        tag_meta_out= os.path.join(args.save_dir, "tag_meta.json")
        cents_path  = os.path.join(args.save_dir, "centroids.pt")
        groups_path = os.path.join(args.save_dir, "group_recs.json")
        tag_upd     = os.path.join(args.save_dir, "tag_recs_grouped.json")
        groups_rep  = os.path.join(args.save_dir, "group_recs_representative.json")

        print(">>> Generating tags ...")
        tag_recs = generate_tag(keys, model_name=args.model)
        _save_json(tag_recs, tag_path)

        print(">>> Embedding (batched) ...")
        key_emb = embed_tags(
            tag_records=tag_recs,
            save_path_keys=key_emb_out,
            embed_model_name=args.embed_model,
            save_path_tags=tag_emb_out,
            save_path_tagmeta=tag_meta_out,
            reduce_to_dim=args.reduce_dim,
            batch_size=args.batch_size
        )
        print("Key emb:", tuple(key_emb.shape), "->", key_emb_out)
        print("Tag emb ->", tag_emb_out)

        print(">>> Grouping per-tag ...")
        tag_emb = _load_tensor(tag_emb_out)
        tag_recs, cents, group_recs = get_tag_group(tag_recs, tag_emb, n_group=args.n_group)
        _save_tensor(cents, cents_path)
        _save_json(group_recs, groups_path)
        _save_json(tag_recs, tag_upd)
        print("Centroids:", tuple(cents.shape), "->", cents_path)

        print(">>> Representative tags ...")
        group_recs = generate_representative_tag(tag_recs, group_recs, tag_emb, n_tag_sample=args.n_tag_sample, model_name=args.model)
        _save_json(group_recs, groups_rep)
        print("representative group_records ->", groups_rep)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
