"""Pairwise judging for advanced DPO batches."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .utils import extract_int, extract_text

__all__ = ["judge_adpo_pairs"]

RequestFunc = Any  # Callable returning list of outputs; left generic on purpose

_SYSTEM_PROMPT = (
    "You are a brilliant judge who decides which text is more relevant to a given query.\n"
    "You will be given a query, 2 sentences.\n"
    "Please carefully analyze these two sentences and then return your answer following the output format.\n\n"
    "Output format:\n<ID> 1 or 2 (file id more relevant to given query) </ID>"
)

_USER_TEMPLATE = (
    "<query>\n{query}\n</query>\n"
    "<sentence id='1'>\n{sentence1}\n</sentence>\n"
    "<sentence id='2'>\n{sentence2}\n</sentence>\n"
)


def _build_prompt(tokenizer, query: str, sentence1: str, sentence2: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(query=query, sentence1=sentence1, sentence2=sentence2)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _maybe_run_async(result):
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


def _load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _format_prompt(query: str, key: str) -> str:
    return (
        "Give me relevant score between query and sentence;\n\n"
        f"Query:{query}\n\n"
        f"Sentence:```{key}```"
    )


def _normalise_key_id(raw_id: Any, fallback: int) -> int:
    if raw_id is None:
        return fallback
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        return fallback


def _load_query_key_set(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, found {type(data).__name__}")

    processed: List[Dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        query = entry.get("query")
        if not query:
            continue

        correspond_keys = entry.get("correspond_keys") or []
        correspond_ids = entry.get("correspond_key_ids") or []
        sampled_keys = entry.get("sampled_keys") or []
        sampled_ids = entry.get("sampled_key_ids") or []

        keys: List[Dict[str, Any]] = []
        for idx, raw_text in enumerate(correspond_keys):
            if raw_text is None:
                continue
            fallback = len(keys)
            raw_id = correspond_ids[idx] if idx < len(correspond_ids) else fallback
            key_id = _normalise_key_id(raw_id, fallback)
            keys.append(
                {
                    "key": str(raw_text),
                    "key_id": key_id,
                    "pair_group": "correspond",
                }
            )

        for idx, raw_text in enumerate(sampled_keys):
            if raw_text is None:
                continue
            fallback = len(keys)
            raw_id = sampled_ids[idx] if idx < len(sampled_ids) else fallback
            key_id = _normalise_key_id(raw_id, fallback)
            keys.append(
                {
                    "key": str(raw_text),
                    "key_id": key_id,
                    "pair_group": "sampled",
                }
            )

        if len(keys) < 2:
            continue

        processed.append(
            {
                "query": query,
                "query_id": entry.get("query_id"),
                "query_type": entry.get("query-type") or entry.get("query_type"),
                "keys": keys,
            }
        )

    if not processed:
        raise ValueError(f"No valid query/key rows found in {path}")
    return processed


def judge_adpo_pairs(
    query_entries: Sequence[Dict[str, Any]],
    *,
    tokenizer: Any | None = None,
    request_func: Optional[RequestFunc] = None,
    batch_size: int = 8,
    sampling_config: Optional[Dict[str, Any]] = None,
    timeout_s: Optional[float] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    progress_dir: Optional[str] = None,
    restart: bool = False,
    sample_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Collect pairwise judgements for ADPO batches."""

    engine_kwargs = engine_kwargs or {}

    tokenizer_source = engine_kwargs.get("tokenizer_name") or engine_kwargs.get("model_name") if engine_kwargs else None
    if request_func is None and tokenizer is None:
        if not tokenizer_source:
            raise ValueError("Provide tokenizer or include 'model_name' / 'tokenizer_name' in engine_kwargs when request_func is omitted")
        tokenizer = _load_tokenizer(tokenizer_source)

    if tokenizer is None:
        raise ValueError("tokenizer must be supplied when request_func is provided")

    requests: List[Dict[str, Any]] = []
    for entry_index, entry in enumerate(query_entries):
        query = entry["query"]
        keys = entry.get("keys", [])
        if len(keys) < 2:
            continue

        all_pairs = list(itertools.combinations(range(len(keys)), 2))
        if sample_pairs is not None and sample_pairs > 0:
            chosen_pairs = random.sample(all_pairs, min(sample_pairs, len(all_pairs)))
        else:
            chosen_pairs = all_pairs

        for idx_a, idx_b in chosen_pairs:
            key_a = keys[idx_a]
            key_b = keys[idx_b]
            prompt = _build_prompt(tokenizer, query, key_a["key"], key_b["key"])
            requests.append(
                {
                    "request_id": len(requests),
                    "entry_index": entry_index,
                    "pair_indices": [idx_a, idx_b],
                    "key_ids": [key_a.get("key_id"), key_b.get("key_id")],
                    "prompt": prompt,
                    "query": query,
                    "query_id": entry.get("query_id"),
                    "query_type": entry.get("query_type"),
                    "keys": [key_a["key"], key_b["key"]],
                }
            )

    existing_results: List[Dict[str, Any]] = []
    finished_ids: Set[int] = set()
    results_path: Optional[Path] = None
    if progress_dir:
        progress_path = Path(progress_dir)
        progress_path.mkdir(parents=True, exist_ok=True)
        results_path = progress_path / "results.json"

        if restart and results_path.exists():
            existing_results = json.loads(results_path.read_text(encoding="utf-8"))
            for record in existing_results:
                rid = record.get("request_id")
                if isinstance(rid, int):
                    finished_ids.add(rid)
    elif restart:
        raise ValueError("Cannot use --restart without specifying --progress-dir.")

    pending_requests = [req for req in requests if req["request_id"] not in finished_ids]

    new_results: List[Dict[str, Any]] = []
    if pending_requests:
        if request_func is None:
            from ..utils import vllm_generate as _vllm_generate
            from vllm import SamplingParams

            engine_params = dict(engine_kwargs)
            model_name = engine_params.pop("model_name", None)
            if not model_name:
                raise ValueError("engine_kwargs must include 'model_name'")

            tensor_parallel_size = engine_params.pop("tensor_parallel_size", 1)
            num_instances = engine_params.pop("num_instances", 1)
            engine_params.pop("tokenizer_name", None)

            sampling_values: Dict[str, Any] = {"temperature": 0.0, "max_tokens": 3000, "top_p": 0.95}
            if sampling_config:
                sampling_values.update(sampling_config)
            sampling = SamplingParams(**sampling_values)

            llm = _vllm_generate.build_llm(
                model_name=model_name,
                tensor_parallel_size=tensor_parallel_size,
                num_instances=num_instances,
                **engine_params,
            )
            outputs_texts: Optional[List[str]] = None
            try:
                outputs_texts = _vllm_generate.generate(
                    llm,
                    [req["prompt"] for req in pending_requests],
                    sampling_params=sampling,
                    batch_size=batch_size,
                    timeout_s=timeout_s,
                )
            finally:
                llm.close(kill=outputs_texts is None)

            if outputs_texts is None:
                raise RuntimeError("vLLM generation failed with no outputs")

            for meta, output_text in zip(pending_requests, outputs_texts):
                record = dict(meta)
                record["output"] = output_text
                new_results.append(record)
        else:
            prompts = [req["prompt"] for req in pending_requests]
            outputs = _maybe_run_async(request_func(prompts))
            for meta, output_text in zip(pending_requests, outputs):
                record = dict(meta)
                record["output"] = output_text
                new_results.append(record)

    results = list(existing_results) + new_results
    results.sort(key=lambda item: item.get("request_id", 0))
    if results_path:
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    return results


def _assemble_dataset_list(
    query_entries: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in results:
        entry_index = record.get("entry_index")
        if entry_index is None:
            continue
        try:
            grouped[int(entry_index)].append(record)
        except (TypeError, ValueError):
            continue

    dataset_list: List[Dict[str, Any]] = []
    for entry_index, entry in enumerate(query_entries):
        keys = entry.get("keys", [])
        if len(keys) < 2:
            continue

        batch: List[Dict[str, Any]] = []
        for position, key_dict in enumerate(keys):
            item: Dict[str, Any] = {
                "msg": [{"role": "user", "content": _format_prompt(entry["query"], key_dict["key"])}],
                "query_id": entry.get("query_id"),
                "key_id": key_dict.get("key_id"),
            }
            if entry.get("query_type"):
                item["query_type"] = entry["query_type"]
            if key_dict.get("pair_group"):
                item["key_role"] = key_dict["pair_group"]
            batch.append(item)

        pair_records = grouped.get(entry_index, [])
        seen_pairs: Set[Tuple[int, int]] = set()
        for record in pair_records:
            pair_indices = record.get("pair_indices") or []
            if len(pair_indices) != 2:
                continue
            output_text = record.get("output", "")
            chosen_id = extract_text(output_text, "ID")
            if chosen_id is None:
                chosen_id = extract_int(output_text[-10:])
            try:
                chosen_val = int(chosen_id)
            except Exception:
                continue
            if chosen_val not in (1, 2):
                continue
            chosen_pos = pair_indices[0] if chosen_val == 1 else pair_indices[1]
            rejected_pos = pair_indices[1] if chosen_val == 1 else pair_indices[0]
            if chosen_pos >= len(keys) or rejected_pos >= len(keys):
                continue
            pair_tuple = (chosen_pos, rejected_pos)
            seen_pairs.add(pair_tuple)

        if not seen_pairs:
            continue

        dataset_list.append(
            {
                "batch": batch,
                "dpo_pairs": [[c, r] for c, r in sorted(seen_pairs)],
            }
        )

    return dataset_list


def main():
    parser = argparse.ArgumentParser(
        description="Collect pairwise relevance judgements for advanced DPO datasets."
    )
    parser.add_argument(
        "--query-key-set",
        "--query-key-s",
        dest="query_key_set",
        type=Path,
        required=True,
        help="JSON generated by sample_advanced_dpo_batch (query/key pairs).",
    )
    parser.add_argument("--model-name", type=str, required=True, help="Local vLLM model path or identifier.")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Optional tokenizer name; defaults to --model-name.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of tensor-parallel shards per worker.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of worker processes to launch.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilisation passed to vLLM.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional maximum context length.")
    parser.add_argument("--dtype", type=str, default=None, help="Optional dtype override for the vLLM engine.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code when loading from HF Hub.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of prompts per vLLM batch.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling value.")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Maximum tokens generated per prompt.")
    parser.add_argument("--timeout-s", type=float, default=None, help="Optional timeout (seconds) for the overall job.")
    parser.add_argument("--progress-dir", type=str, default=None, help="Optional directory for progress checkpoints.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_list.json"),
        help="Destination path for the assembled dataset list JSON.",
    )
    parser.add_argument("--restart", action="store_true", help="Resume from existing progress logs if available.")
    parser.add_argument(
        "--sample-pairs",
        type=int,
        default=0,
        help="Number of sentence pairs sampled per query (set to 0 to use all pairs).",
    )
    args = parser.parse_args()

    query_entries = _load_query_key_set(args.query_key_set)

    engine_kwargs = {
        "model_name": args.model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "num_instances": args.num_instances,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.tokenizer_name:
        engine_kwargs["tokenizer_name"] = args.tokenizer_name
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    if args.dtype:
        engine_kwargs["dtype"] = args.dtype
    if args.trust_remote_code:
        engine_kwargs["trust_remote_code"] = True

    sample_pairs = args.sample_pairs if args.sample_pairs and args.sample_pairs > 0 else None

    results = judge_adpo_pairs(
        query_entries,
        tokenizer=None,
        request_func=None,
        batch_size=args.batch_size,
        sampling_config={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        timeout_s=args.timeout_s,
        engine_kwargs=engine_kwargs,
        progress_dir=args.progress_dir,
        restart=args.restart,
        sample_pairs=sample_pairs,
    )

    if args.progress_dir:
        progress_path = Path(args.progress_dir) / "results.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        saved_progress = f"Saved {len(results)} judgements to {progress_path}"
    else:
        saved_progress = f"Processed {len(results)} judgements (no progress directory specified)"

    dataset_list = _assemble_dataset_list(query_entries, results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset_list, ensure_ascii=False, indent=2))
    print(saved_progress)
    print(f"Wrote dataset list with {len(dataset_list)} entries to {args.output}")


if __name__ == "__main__":
    main()
