# tag_pipeline.py
from __future__ import annotations
import os, json, random, math, re
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
    Falls back to extracting lines after markers or quoted items.
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
    # Keep config minimal; customize as you like
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_mem_util,
    )
    return llm


def _completion_json_list(llm: LLM, prompts: List[str], max_tags_per_key=6) -> List[List[str]]:
    """
    Run vLLM generation batched; return list of tag lists (per prompt).
    """
    # Lower temperature for more consistent JSON-ish output
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
            t = re.sub(r"[^\w\-&/ +]", "", t).strip()  # keep it clean but allow common separators
            t = re.sub(r"\s+", " ", t)
            if not t or t.lower() in seen:
                continue
            seen.add(t.lower())
            clean.append(t)
            if len(clean) >= max_tags_per_key:
                break
        if not clean:
            clean = ["general"]  # minimal safe fallback
        all_tags.append(clean)
    return all_tags


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


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

    # Prompt template: force JSON list of short tags
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
    # vLLM generates efficiently in batches; adjust batch size to your VRAM
    for batch in _batched(prompts, bs=64):
        tag_lists.extend(_completion_json_list(llm, batch))

    tag_records = []
    for i, (k, tags) in enumerate(zip(keys, tag_lists)):
        tag_records.append({
            "key": k,
            "key_id": i,
            "tags": tags,              # e.g., ["Graph RAG", "Reward Model", ...]
        })
    return tag_records


# ----------------------------- #
# 2) Embed tags with SentenceTransformer (-> per-key embedding)
# ----------------------------- #
@torch.no_grad()
def embed_tags(
    tag_records: List[Dict[str, Any]],
    save_path: Optional[str],
    embed_model_name: str,
    quantize_precision: Optional[str] = None,   # e.g., "ubinary"
    reduce_to_dim: Optional[int] = None,        # e.g., 512 (optional PCA-lite)
) -> torch.Tensor:
    """
    Computes an embedding per key by averaging its tag embeddings.
    Returns: embeddings: torch.FloatTensor [n_keys, embed_dim]
    """
    if not tag_records:
        return torch.empty(0, 0)

    device = _device()
    model = SentenceTransformer(embed_model_name).to(device)

    # Flatten all tags, track (key_id, tag_idx)
    flat_tags: List[str] = []
    tag_meta: List[Tuple[int, int]] = []
    for rec in tag_records:
        for j, t in enumerate(rec["tags"]):
            flat_tags.append(t)
            tag_meta.append((rec["key_id"], j))

    # Encode all tags in batches (SentenceTransformer handles batching)
    tag_emb = model.encode(flat_tags, convert_to_tensor=True, device=device, normalize_embeddings=True)
    # tag_emb: [n_tags_total, dim]

    # Optional post-quantization (binary etc.)
    if quantize_precision:
        q = quantize_embeddings(tag_emb, precision=quantize_precision)
        tag_emb = torch.tensor(q, device=device)

    # Optional dimension reduction with torch SVD (PCA-like)
    if reduce_to_dim is not None and reduce_to_dim > 0 and reduce_to_dim < tag_emb.shape[-1]:
        # Centering
        mu = tag_emb.mean(dim=0, keepdim=True)
        X = tag_emb - mu
        # Economy SVD
        # For large matrices, you may want to sample for speed
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        W = Vh[:reduce_to_dim, :].T  # [D, r]
        tag_emb = X @ W
        # Re-normalize after reduction
        tag_emb = _normalize(tag_emb)

    dim = tag_emb.shape[-1]

    # Aggregate to per-key embedding (mean over that key's tags)
    n_keys = len(tag_records)
    key_emb = torch.zeros(n_keys, dim, device=tag_emb.device, dtype=tag_emb.dtype)
    counts = torch.zeros(n_keys, device=tag_emb.device)
    for idx, (kid, _tagid) in enumerate(tag_meta):
        key_emb[kid] += tag_emb[idx]
        counts[kid] += 1
    counts = counts.clamp_min(1.0).unsqueeze(-1)
    key_emb = key_emb / counts
    key_emb = _normalize(key_emb).contiguous()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(key_emb, save_path)

    return key_emb


# ----------------------------- #
# 3) Group keys by embeddings (Torch k-means)
# ----------------------------- #
@torch.no_grad()
def get_tag_group(
    tag_records: List[Dict[str, Any]],
    embeddings: torch.Tensor,
    n_group: int,
    n_init: int = 1,
    max_iters: int = 50,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Assign each key to ONE cluster. We still store as a list in 'group_ids' for extensibility.

    Returns:
      - tag_records (updated with "group_ids": [int])
      - group_records: [
           {
             "group_id": int,
             "centroid": torch.FloatTensor [dim],
             "tags": List[str],                 # union of member keys' tags (deduped)
             "tag_ids": List[Tuple[int,int]],   # (key_id, tag_index)
           }, ...
        ]
    """
    if embeddings.numel() == 0 or n_group <= 0:
        # trivial return
        for rec in tag_records:
            rec["group_ids"] = []
        return tag_records, []

    torch.manual_seed(seed)
    X = embeddings.detach()  # [N, D]
    N, D = X.shape
    n_group = max(1, min(n_group, N))

    def kmeans_run() -> Tuple[torch.Tensor, torch.Tensor]:
        # k-means++ init
        centroids = torch.empty(n_group, D, device=X.device, dtype=X.dtype)
        # pick first center
        idx0 = torch.randint(0, N, (1,), device=X.device)
        centroids[0] = X[idx0]
        # pick others
        closest_dist_sq = torch.cdist(X, centroids[:1]).squeeze(-1) ** 2  # [N]
        for c in range(1, n_group):
            probs = (closest_dist_sq / closest_dist_sq.sum()).clamp_min(1e-12)
            choice = torch.multinomial(probs, 1)
            centroids[c] = X[choice]
            dist_sq = torch.cdist(X, centroids[c:c+1]).squeeze(-1) ** 2
            closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq)

        prev_assign = torch.full((N,), -1, device=X.device, dtype=torch.long)
        for it in range(max_iters):
            # Assign
            d = torch.cdist(X, centroids)  # [N, K]
            assign = torch.argmin(d, dim=1)  # [N]
            if torch.equal(assign, prev_assign):
                break
            prev_assign = assign
            # Update
            for k in range(n_group):
                mask = (assign == k)
                if mask.any():
                    centroids[k] = X[mask].mean(dim=0)
                # else: keep centroid
        return assign, centroids

    best_inertia = float("inf")
    best_assign = None
    best_centroids = None
    for _ in range(n_init):
        assign, cents = kmeans_run()
        # inertia
        inertia = (X - cents[assign]).pow(2).sum().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_assign = assign
            best_centroids = cents

    assign = best_assign
    centroids = _normalize(best_centroids)

    # Update tag_records with group_ids
    for rec in tag_records:
        gid = int(assign[rec["key_id"]].item())
        rec["group_ids"] = [gid]

    # Build group_records (union of tags + tag_ids)
    groups: List[Dict[str, Any]] = []
    for gid in range(n_group):
        groups.append({
            "group_id": gid,
            "centroid": centroids[gid].detach().clone(),
            "tags": [],
            "tag_ids": [],
        })

    seen_per_group = [set() for _ in range(n_group)]
    for rec in tag_records:
        gid = rec["group_ids"][0]
        for j, t in enumerate(rec["tags"]):
            key = t.lower().strip()
            if key not in seen_per_group[gid]:
                seen_per_group[gid].add(key)
                groups[gid]["tags"].append(t)
                groups[gid]["tag_ids"].append((rec["key_id"], j))

    return tag_records, groups


# ----------------------------- #
# 4) Representative tag per group (vLLM)
# ----------------------------- #
def generate_representative_tag(
    tag_records: List[Dict[str, Any]],
    group_records: List[Dict[str, Any]],
    embeddings: torch.Tensor,
    n_tag_sample: int,
    model_name: str
) -> List[Dict[str, Any]]:
    """
    For each group, sample up to n_tag_sample of its tags, then ask the LLM
    for ONE short representative tag/label.
    Adds "representative_tag" to each group record.
    """
    if not group_records:
        return group_records

    llm = _build_llm(model_name)
    prompts = []
    for g in group_records:
        tags = g["tags"]
        if not tags:
            sample = ["general"]
        else:
            if len(tags) <= n_tag_sample:
                sample = tags
            else:
                sample = random.sample(tags, n_tag_sample)

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
        # clean
        txt = re.sub(r"[\n\r\"'`]", " ", txt).strip()
        txt = re.sub(r"\s+", " ", txt)
        g["representative_tag"] = txt[:60] if txt else "General"
    return group_records


# ----------------------------- #
# Debug / Example usage
# ----------------------------- #
if __name__ == "__main__":
    """
    Minimal debugging run. Adjust model names to ones you have locally.
    - For vLLM (generation): use a small instruct model, e.g. "Qwen2.5-3B-Instruct" or "meta-llama/Meta-Llama-3-8B-Instruct".
    - For embeddings: the user’s reference model "intfloat/e5-mistral-7b-instruct" works with SentenceTransformer.
    """
    # ----- CONFIG -----
    data_name = "smollm-corpus"
    GEN_MODEL = os.environ.get("VLLM_MODEL", "/workspace/qwen7b")
    EMB_MODEL = os.environ.get("EMB_MODEL", "intfloat/e5-mistral-7b-instruct")
    SAVE_EMB = f"/workspace/RMS_exp/data/{data_name}/key_embeddings.pt"
    N_GROUP = 4
    N_TAG_SAMPLE = 6
    random.seed(0)

    # ----- SAMPLE KEYS (replace with yours) -----
    sample_keys = [
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

    print(">>> Generating tags with vLLM ...")
    tag_recs = generate_tag(sample_keys, model_name=GEN_MODEL)
    for r in tag_recs[:2]:
        print(f"[key_id={r['key_id']}] {r['key']}\n  tags={r['tags']}")

    print("\n>>> Embedding tags with SentenceTransformer ...")
    emb = embed_tags(
        tag_records=tag_recs,
        save_path=SAVE_EMB,
        embed_model_name=EMB_MODEL,
        quantize_precision=None,   # or "ubinary"
        reduce_to_dim=512          # optional; set None to keep original dim
    )
    print("embeddings:", tuple(emb.shape), emb.dtype, emb.device)

    print("\n>>> Clustering keys into groups ...")
    tag_recs, group_recs = get_tag_group(tag_records=tag_recs, embeddings=emb, n_group=N_GROUP)
    for g in group_recs:
        print(f"[group {g['group_id']}] members≈{len(g['tag_ids'])} tags_in_union={len(g['tags'])}")

    print("\n>>> Generating representative tag per group with vLLM ...")
    group_recs = generate_representative_tag(
        tag_records=tag_recs,
        group_records=group_recs,
        embeddings=emb,
        n_tag_sample=N_TAG_SAMPLE,
        model_name=GEN_MODEL
    )
    for g in group_recs:
        print(f"[group {g['group_id']}] rep='{g['representative_tag']}'")

    print("\n>>> Done. Saved key embeddings to:", SAVE_EMB)
