"""Generate per-key query metadata using an LLM backend."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .utils import extract_text

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


logger = logging.getLogger(__name__)


def _generate_stub_queries(texts: Sequence[str], *, random_seed: int = 42) -> Dict[int, Dict[str, List[str]]]:
    """Generate deterministic placeholder queries when the real model is unavailable."""

    rng = random.Random(random_seed)
    fallback_irrelevant_pool = [
        "What is your favourite colour?",
        "Which sport do you like the most?",
        "What did you eat for breakfast?",
        "Where would you travel for fun?",
    ]

    query_dict: Dict[int, Dict[str, List[str]]] = {}
    for idx, sentence in enumerate(texts):
        tokens = re.findall(r"[A-Za-z0-9]+", sentence)
        unique_tokens: List[str] = []
        for token in tokens:
            lower = token.lower()
            if lower not in unique_tokens:
                unique_tokens.append(lower)
        keywords = unique_tokens[:5] if unique_tokens else ["topic"]

        clean_sentence = " ".join(sentence.strip().split())
        if len(clean_sentence) > 120:
            title = clean_sentence[:117].rstrip() + "..."
        else:
            title = clean_sentence or "Untitled sentence"

        question = f"What is the main idea of: {title}?"
        irr_question = rng.choice(fallback_irrelevant_pool)

        query_dict[idx] = {
            "titles": [title],
            "keywords": keywords,
            "questions": [question],
            "irr_questions": [irr_question],
        }

    return query_dict


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


def _load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
    batch_size: int = 8,
    sampling_config: Optional[Dict[str, Any]] = None,
    timeout_s: Optional[float] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, List[str]]]:
    """Build query metadata for each input text.

    Returns ``{request_id: {"titles": [...], "keywords": [...], "questions": [...], "irr_questions": [...]}}``.
    """

    engine_kwargs = engine_kwargs or {}

    tokenizer_source = engine_kwargs.get("tokenizer_name") or engine_kwargs.get("model_name") if engine_kwargs else None
    if request_func is None and tokenizer is None:
        if not tokenizer_source:
            raise ValueError("Provide tokenizer or include 'model_name' / 'tokenizer_name' in engine_kwargs when request_func is omitted")
        try:
            tokenizer = _load_tokenizer(tokenizer_source)
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            logger.warning("Falling back to stub query generation because the tokenizer failed to load: %s", exc)
            return _generate_stub_queries(texts)

    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    build_prompt = _default_prompt_builder(tokenizer)
    prompts = [build_prompt(text) for text in texts]

    outputs: List[Tuple[int, str]]
    if request_func is None:
        if engine_kwargs is None:
            raise ValueError("engine_kwargs must be provided when request_func is omitted")

        try:
            from ..utils import vllm_generate as _vllm_generate
            from vllm import SamplingParams
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            logger.warning("Falling back to stub query generation because vLLM could not be imported: %s", exc)
            return _generate_stub_queries(texts)

        engine_params = dict(engine_kwargs)
        model_name = engine_params.pop("model_name", None)
        if not model_name:
            raise ValueError("engine_kwargs must include 'model_name'")

        tensor_parallel_size = engine_params.pop("tensor_parallel_size", 1)
        num_instances = engine_params.pop("num_instances", 1)
        engine_params.pop("tokenizer_name", None)  # already handled above

        sampling_values: Dict[str, Any] = {"temperature": 0.0, "max_tokens": 3000, "top_p": 0.95}
        if sampling_config:
            sampling_values.update(sampling_config)
        sampling = SamplingParams(**sampling_values)

        llm = None
        try:
            llm = _vllm_generate.build_llm(
                model_name=model_name,
                tensor_parallel_size=tensor_parallel_size,
                num_instances=num_instances,
                **engine_params,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            logger.warning("Falling back to stub query generation because the vLLM worker failed to start: %s", exc)
            return _generate_stub_queries(texts)

        outputs_texts: Optional[List[str]] = None
        try:
            outputs_texts = _vllm_generate.generate(
                llm,
                prompts,
                sampling_params=sampling,
                batch_size=batch_size,
                timeout_s=timeout_s,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            logger.warning("vLLM generation failed (%s); using stub outputs instead.", exc)
        finally:
            if llm is not None:
                llm.close(kill=outputs_texts is None)

        if outputs_texts is None:
            return _generate_stub_queries(texts)

        outputs = list(enumerate(outputs_texts))
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

    # query_dict (dict): maps integer request ids to
    #   {
    #     "titles": ["<generated title>", ...],
    #     "keywords": ["<keyword>", ...],
    #     "questions": ["<question>", ...],
    #     "irr_questions": ["<irrelevant question>", ...]
    #   }
    return query_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions, titles, and keywords for source sentences.")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV file containing source texts.")
    parser.add_argument("--text-column", type=str, default="text", help="Column containing the text to analyse.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file for generated queries.")
    parser.add_argument("--model-name", type=str, required=True, help="Local vLLM model path or identifier.")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Optional tokenizer name; defaults to --model-name.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of tensor parallel shards per worker.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of worker processes to launch.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilisation passed to vLLM.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional maximum model context length.")
    parser.add_argument("--dtype", type=str, default=None, help="Optional dtype override for the vLLM engine.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code when loading from HF Hub.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of prompts sent per generation batch.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling value.")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Maximum tokens generated per prompt.")
    parser.add_argument("--timeout-s", type=float, default=None, help="Optional timeout (in seconds) for the overall job.")
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in {args.input_csv}")
    sentences = df[args.text_column].dropna().astype(str).tolist()
    if not sentences:
        raise ValueError("No sentences available for query generation.")

    engine_kwargs = {
        "model_name": args.model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "num_instances": args.num_instances,
    }
    if args.tokenizer_name:
        engine_kwargs["tokenizer_name"] = args.tokenizer_name
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    if args.dtype:
        engine_kwargs["dtype"] = args.dtype
    if args.trust_remote_code:
        engine_kwargs["trust_remote_code"] = True

    query_dict = make_queries(
        sentences,
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
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(query_dict, ensure_ascii=False, indent=2))
    print(f"Saved generated queries to {args.output}")
