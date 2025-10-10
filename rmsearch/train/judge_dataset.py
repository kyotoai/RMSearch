"""Pairwise sentence judging helpers."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from .utils import extract_int, extract_text

__all__ = ["judge_sentences"]

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


def judge_sentences(
    relevant_sentences: Sequence[Dict[str, Any]],
    *,
    tokenizer: Any | None = None,
    request_func: Optional[RequestFunc] = None,
    batch_size: int = 8,
    sampling_config: Optional[Dict[str, Any]] = None,
    timeout_s: Optional[float] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    progress_dir: Optional[str] = None,
    restart: bool = False,
    sample_pairs: int = 1,
) -> List[Dict[str, Any]]:
    """Request pairwise judgements for candidate sentences.

    ``relevant_sentences`` structure -> ``[{"query_id": int, "query": str, "keys": [{"key_id": int, "key": str}, ...]}]``.
    """

    engine_kwargs = engine_kwargs or {}

    tokenizer_source = engine_kwargs.get("tokenizer_name") or engine_kwargs.get("model_name") if engine_kwargs else None
    if request_func is None and tokenizer is None:
        if not tokenizer_source:
            raise ValueError("Provide tokenizer or include 'model_name' / 'tokenizer_name' in engine_kwargs when request_func is omitted")
        tokenizer = _load_tokenizer(tokenizer_source)

    if tokenizer is None:
        raise ValueError("tokenizer must be supplied when request_func is provided")

    requests: List[Dict[str, Any]] = []

    for sentence_dict in relevant_sentences:
        query = sentence_dict["query"]
        keys = sentence_dict.get("keys", [])
        sentence_ids = [int(item["key_id"]) for item in keys]
        sentences = [str(item["key"]) for item in keys]

        sentence_pairs = list(itertools.combinations(range(len(sentence_ids)), 2))
        if not sentence_pairs:
            continue
        chosen_pairs = random.sample(sentence_pairs, min(sample_pairs, len(sentence_pairs)))

        for idx_a, idx_b in chosen_pairs:
            sent_id1 = sentence_ids[idx_a]
            sent_id2 = sentence_ids[idx_b]
            prompt = _build_prompt(tokenizer, query, sentences[idx_a], sentences[idx_b])
            request = {
                "request_id": len(requests),
                "prompt": prompt,
                "sentence_ids": [sent_id1, sent_id2],
                "question": query,
                "query_id": sentence_dict.get("query_id"),
                "query_type": sentence_dict.get("query_type") or sentence_dict.get("query-type"),
                "sentences": [sentences[idx_a], sentences[idx_b]],
            }
            requests.append(request)

    existing_results: List[Dict[str, Any]] = []
    finished_ids: Set[int] = set()
    results_path: Optional[Path] = None
    if progress_dir:
        progress_path = Path(progress_dir)
        progress_path.mkdir(parents=True, exist_ok=True)
        results_path = progress_path / "results.json"

        if restart and results_path.exists():
            existing_results = json.loads(results_path.read_text())
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

    # results (list): records like
    #   {
    #     "request_id": <sequential id>,
    #     "prompt": "<full prompt sent to the model>",
    #     "sentence_ids": [<id_a>, <id_b>],
    #     "question": "<original query text>",
    #     "output": "<model judgement containing <ID> tag>"
    #   }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect pairwise relevance judgements for candidate sentences.")
    parser.add_argument(
        "--query-key-set",
        "--query-key-s",
        dest="query_key_set",
        type=Path,
        help="JSON file from sample_dpo_batch (query/key pairs).",
    )
    parser.add_argument(
        "--relevant-json",
        type=Path,
        help="Legacy JSON file containing relevant sentences per query.",
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
    parser.add_argument("--sample-pairs", type=int, default=1, help="Number of sentence pairs sampled per query.")
    args = parser.parse_args()

    if not args.query_key_set and not args.relevant_json:
        raise ValueError("Provide --query-key-set (preferred) or --relevant-json.")

    def _build_from_query_key_set(path: Path):
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}, found {type(data).__name__}")
        formatted: List[Dict[str, Any]] = []
        lookup: Dict[tuple[str, int], str] = {}
        query_meta: Dict[str, Dict[str, Any]] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            query = entry.get("query")
            if not query:
                continue
            keys = entry.get("keys") or []
            key_ids = entry.get("key_ids") or []
            key_payload: List[Dict[str, Any]] = []
            for idx, raw_text in enumerate(keys):
                if raw_text is None:
                    continue
                try:
                    key_id = int(key_ids[idx]) if idx < len(key_ids) else idx
                except (TypeError, ValueError):
                    key_id = idx
                text = str(raw_text)
                key_payload.append({"key_id": key_id, "key": text})
                lookup[(query, key_id)] = text
            if len(key_payload) < 2:
                continue
            query_type = entry.get("query-type") or entry.get("query_type")
            formatted.append(
                {
                    "query": query,
                    "query_id": entry.get("query_id"),
                    "query_type": query_type,
                    "keys": key_payload,
                }
            )
            query_meta[query] = {
                "query_id": entry.get("query_id"),
                "query_type": query_type,
            }
        if not formatted:
            raise ValueError(f"No valid query/key pairs found in {path}")
        return formatted, lookup, query_meta

    def _build_from_relevant_json(path: Path):
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}, found {type(data).__name__}")
        lookup: Dict[tuple[str, int], str] = {}
        query_meta: Dict[str, Dict[str, Any]] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            query = entry.get("query")
            if not query:
                continue
            for item in entry.get("keys", []):
                if isinstance(item, dict) and "key_id" in item and "key" in item:
                    try:
                        key_id = int(item["key_id"])
                    except (TypeError, ValueError):
                        continue
                    lookup[(query, key_id)] = str(item["key"])
        return data, lookup, query_meta

    if args.query_key_set:
        relevant_sentences, text_lookup, query_meta = _build_from_query_key_set(args.query_key_set)
    else:
        if not args.relevant_json or not args.relevant_json.exists():
            raise FileNotFoundError("Relevant sentences file not found.")
        relevant_sentences, text_lookup, query_meta = _build_from_relevant_json(args.relevant_json)

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

    results = judge_sentences(
        relevant_sentences,
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
        sample_pairs=args.sample_pairs,
    )

    if args.progress_dir:
        progress_path = Path(args.progress_dir) / "results.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        saved_progress = f"Saved {len(results)} judgements to {progress_path}"
    else:
        saved_progress = f"Processed {len(results)} judgements (no progress directory specified)"

    dataset_list: List[Dict[str, Any]] = []
    for result in results:
        question = result.get("question")
        if not question:
            continue
        sentence_ids = result.get("sentence_ids") or []
        sentences = result.get("sentences") or []
        if len(sentence_ids) < 2:
            continue
        output_text = result.get("output", "")
        chosen_id = extract_text(output_text, "ID")
        if chosen_id is None:
            chosen_id = extract_int(output_text[-10:])
        try:
            chosen_val = int(chosen_id)
        except Exception:
            continue
        if chosen_val == 1:
            chosen_idx = 0
        elif chosen_val == 2:
            chosen_idx = 1
        else:
            continue
        other_idx = 1 - chosen_idx
        try:
            chosen_sentence_id = int(sentence_ids[chosen_idx])
            rejected_sentence_id = int(sentence_ids[other_idx])
        except (TypeError, ValueError):
            continue

        def _resolve_text(idx: int, sentence_id: int) -> Optional[str]:
            if idx < len(sentences):
                text_val = sentences[idx]
                if text_val:
                    return str(text_val)
            return text_lookup.get((question, sentence_id))

        chosen_text = _resolve_text(chosen_idx, chosen_sentence_id)
        rejected_text = _resolve_text(other_idx, rejected_sentence_id)
        if not chosen_text or not rejected_text:
            continue

        meta = query_meta.get(question, {})

        def _format_prompt(text: str) -> str:
            return (
                "Give me relevant score between query and sentence;\n\n"
                f"Query:{question}\n\n"
                f"Sentence:```{text}```"
            )

        dataset_list.append(
            {
                "query": question,
                "query_id": result.get("query_id") or meta.get("query_id"),
                "query_type": result.get("query_type") or meta.get("query_type"),
                "chosen_msg": [{"role": "user", "content": _format_prompt(chosen_text)}],
                "rejected_msg": [{"role": "user", "content": _format_prompt(rejected_text)}],
                "chosen_sentence_id": chosen_sentence_id,
                "rejected_sentence_id": rejected_sentence_id,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset_list, ensure_ascii=False, indent=2))
    print(saved_progress)
    print(f"Wrote dataset list with {len(dataset_list)} entries to {args.output}")
