"""Pairwise sentence judging helpers."""

from __future__ import annotations

import asyncio
import itertools
import random
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
    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            del tokenize, add_generation_prompt
            return "\n".join(block["content"] for block in messages)

    rel = [
        {
            "query_id": 0,
            "query": "What is retrieval?",
            "keys": [
                {"key_id": 1, "key": "Retrieval augments generation."},
                {"key_id": 2, "key": "Cooking is fun."},
            ],
        }
    ]

    def fake_request(prompts: List[str]) -> List[str]:
        return ["<ID>1</ID>" for _ in prompts]

    judged = judge_sentences(rel, tokenizer=DummyTokenizer(), request_func=fake_request)
    print(judged)
