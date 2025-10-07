"""Pairwise sentence judging helpers."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .utils import AllRequests, setup_async_engine

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


def judge_sentences(
    relevant_sentences: Sequence[Dict[str, Any]],
    *,
    tokenizer: Any | None = None,
    request_func: Optional[RequestFunc] = None,
    max_requests: int = 40,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    progress_dir: str = "relevant_file_progress",
    restart: bool = False,
    sample_pairs: int = 1,
) -> List[Dict[str, Any]]:
    """Request pairwise judgements for candidate sentences.

    ``relevant_sentences`` structure -> ``[{"query_id": int, "query": str, "keys": [{"key_id": int, "key": str}, ...]}]``.
    """

    engine_kwargs = engine_kwargs or {}

    if request_func is None:
        if tokenizer is None:
            if "model_name" not in engine_kwargs:
                raise ValueError("Provide tokenizer or engine_kwargs['model_name'] when request_func is omitted")
            _, tokenizer = setup_async_engine(**engine_kwargs)
    if tokenizer is None:
        raise ValueError("tokenizer must be supplied when request_func is provided")

    all_requests = AllRequests(max_request=max_requests, engine_kwargs=engine_kwargs)

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
                "request_id": len(all_requests.requests),
                "prompt": prompt,
                "sentence_ids": [sent_id1, sent_id2],
                "question": query,
            }
            all_requests.add(request)

    if request_func is None:
        results = asyncio.run(
            all_requests.process(
                model_name=engine_kwargs.get("model_name"),
                max_tokens=3000,
                temperature=0.0,
                save_dir=progress_dir,
                restart=restart,
            )
        )
    else:
        prompts = [req["prompt"] for req in all_requests.requests]
        outputs = _maybe_run_async(request_func(prompts))
        results = []
        for meta, output_text in zip(all_requests.requests, outputs):
            record = dict(meta)
            record["output"] = output_text
            results.append(record)

    # results structure -> [{"request_id": int, "prompt": str, "sentence_ids": [int, int], "question": str, "output": str}]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect pairwise relevance judgements for candidate sentences.")
    parser.add_argument("--relevant-json", type=Path, required=True, help="JSON file containing relevant sentences per query.")
    parser.add_argument("--model-name", type=str, required=True, help="Async vLLM model used to deliver judgements.")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="tensor_parallel_size for the async engine.")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1, help="pipeline_parallel_size for the async engine.")
    parser.add_argument("--data-parallel-size", type=int, default=1, help="data_parallel_size for the async engine.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="GPU memory utilisation passed to AsyncLLMEngine.")
    parser.add_argument("--omp-num-threads", type=int, default=4, help="Number of CPU threads for the async workers.")
    parser.add_argument("--max-requests", type=int, default=40, help="Maximum concurrent requests issued to the engine.")
    parser.add_argument("--progress-dir", type=str, default="relevant_file_progress", help="Directory for progress checkpoints.")
    parser.add_argument("--output", type=Path, default=None, help="Destination path for the collected judgements JSON.")
    parser.add_argument("--restart", action="store_true", help="Resume from existing progress logs if available.")
    parser.add_argument("--sample-pairs", type=int, default=1, help="Number of sentence pairs sampled per query.")
    args = parser.parse_args()

    if not args.relevant_json.exists():
        raise FileNotFoundError(f"Relevant sentences file not found: {args.relevant_json}")

    relevant_sentences = json.loads(args.relevant_json.read_text())

    engine_kwargs = {
        "model_name": args.model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "omp_num_threads": args.omp_num_threads,
    }

    results = judge_sentences(
        relevant_sentences,
        tokenizer=None,
        request_func=None,
        max_requests=args.max_requests,
        engine_kwargs=engine_kwargs,
        progress_dir=args.progress_dir,
        restart=args.restart,
        sample_pairs=args.sample_pairs,
    )

    output_path = args.output or (Path(args.progress_dir) / "results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved judgements to {output_path}")
