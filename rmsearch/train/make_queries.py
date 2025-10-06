"""Generate per-key query metadata using an LLM backend."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .utils import AllRequests, extract_text, setup_async_engine

__all__ = ["make_queries"]

PromptBuilder = Callable[[str], str]
RequestFunc = Callable[[List[str]], List[str]] | Callable[[List[str]], Awaitable[List[str]]]

_SYSTEM_PROMPT = (
    "You are a helpful assistant who extracts titles, and keywords from a sentence provided by the user,"
    " and also creates questions and irrelevant questions.\n"
    "Following the user's instructions, analyze the content of the sentence and respond according to the output format below.\n"
    "Make sure that your questions are creative and sometimes that asks question\n\n"
    "Output format:\n"
    "<titles>[\"Title1\", \"Title2\", ... ]</titles>\n"
    "<keywords>[\"Keyword1\", \"Keyword2\", ... ]</keywords>\n"
    "<questions>[\"Question1\", \"Question2\", ... ]</questions>\n"
    "<irrelevant questions>[\"Question1\", \"Question2\", ... ]</irrelevant questions>"
)

_USER_TEMPLATE = (
    "Sentence:\n'''\n{sentence}\n'''\n\n"
    "Instructions:\n"
    "1. Summarize the content of the sentence into 2-3 one-line titles.\n"
    "2. Extract 3–5 main keywords from the sentence.\n"
    "3. Create several questions and irrelevant ones about the sentence, ranging from easy to difficult.\n"
    "4. Enclose each element in order with the tags <titles></titles>, <keywords></keywords>,"
    " and <irrelevant questions></irrelevant questions> when outputting.\n\n"
    "Follow the instructions step-by-step and think in sequence."
)


def _default_prompt_builder(tokenizer) -> PromptBuilder:
    def build_prompt(sentence: str) -> str:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(sentence=sentence)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    return build_prompt


def _maybe_run_async(maybe_result):
    if asyncio.iscoroutine(maybe_result):
        return asyncio.run(maybe_result)
    return maybe_result


def _safe_json(text: Optional[str]) -> List[str]:
    if not text:
        return []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return [str(item) for item in loaded if isinstance(item, (str, int, float))]
    except Exception:
        pass
    return []


def make_queries(
    texts: Sequence[str],
    *,
    tokenizer: Any | None = None,
    request_func: Optional[RequestFunc] = None,
    max_requests: int = 50,
    progress_dir: str = "progress_questions",
    restart: bool = False,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, List[str]]]:
    """Build query metadata for each input text.

    Returns ``{request_id: {"titles": [...], "keywords": [...], "questions": [...], "irr_questions": [...]}}``.
    """

    engine_kwargs = engine_kwargs or {}

    if request_func is None and tokenizer is None:
        if "model_name" not in engine_kwargs:
            raise ValueError("Provide tokenizer or engine_kwargs['model_name'] when request_func is omitted")
        _, tokenizer = setup_async_engine(**engine_kwargs)
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    build_prompt = _default_prompt_builder(tokenizer)
    prompts = [build_prompt(text) for text in texts]

    outputs: List[Tuple[int, str]]
    if request_func is None:
        all_requests = AllRequests(max_request=max_requests, engine_kwargs=engine_kwargs)
        for idx, prompt in enumerate(prompts):
            all_requests.add({"request_id": idx, "prompt": prompt})
        results = asyncio.run(
            all_requests.process(
                model_name=engine_kwargs.get("model_name"),
                max_tokens=3000,
                temperature=0.0,
                save_dir=progress_dir,
                restart=restart,
            )
        )
        outputs = [(record.get("request_id", idx), record.get("output", "")) for idx, record in enumerate(results)]
    else:
        responses = _maybe_run_async(request_func(prompts))
        outputs = list(enumerate(responses))

    query_dict: Dict[int, Dict[str, List[str]]] = {}

    for request_id, raw_output in outputs:
        titles = extract_text(raw_output, "titles")
        keywords = extract_text(raw_output, "keywords")
        questions = extract_text(raw_output, "questions")
        irr_questions = extract_text(raw_output, "irrelevant questions")

        query_dict[request_id] = {
            "titles": _safe_json(titles),
            "keywords": _safe_json(keywords),
            "questions": _safe_json(questions),
            "irr_questions": _safe_json(irr_questions),
        }

    # query_dict structure -> {request_id: {"titles": [...], "keywords": [...], "questions": [...], "irr_questions": [...]}}
    return query_dict


if __name__ == "__main__":
    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            del tokenize, add_generation_prompt
            combined = "\n".join(block["content"] for block in messages)
            return combined

    def fake_request(prompts: List[str]) -> List[str]:
        return [
            "<titles>[\"T1\"]</titles><keywords>[\"K1\"]</keywords><questions>[\"Q1\"]</questions><irrelevant questions>[\"I1\"]</irrelevant questions>"
            for _ in prompts
        ]

    sample_queries = make_queries(["Example sentence."], tokenizer=DummyTokenizer(), request_func=fake_request)
    print(sample_queries)
