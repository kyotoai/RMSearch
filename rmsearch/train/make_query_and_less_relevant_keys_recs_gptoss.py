"""Generate query and ordered less-relevant key records with an LLM backend."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

PromptBuilder = Callable[[str, "InstructionPair"], str]
RequestFunc = Callable[[List[str]], List[str]] | Callable[[List[str]], Awaitable[List[str]]]

_SYSTEM_PROMPT = (
    "You are a meticulous data generation assistant building retrieval training corpora.\n"
    "Follow the user's recipe exactly, keep every item grounded in the source key, and return strictly valid JSON."
)

_USER_TEMPLATE = (
    "Original key (keep confidential, do not echo verbatim):\n"
    "'''\n"
    "{sentence}\n"
    "'''\n\n"
    "Query style [{query_instruction_id}]: {query_instruction}\n"
    "Less-relevant strategy [{less_instruction_id}]: {less_instruction}\n\n"
    "Tasks:\n"
    "1. Draft ONE query that follows the query style instructions.\n"
    "2. Produce EXACTLY {n_keys} less_relevant_keys. Index 0 must be almost as relevant as the original key, "
    "and each subsequent index must become gradually less relevant while remaining on-topic.\n"
    "3. Keep the format, tone, and length of each less_relevant_key within ±10% of the original key's word count and match its paragraph or bullet structure so they are hard to distinguish on appearance alone.\n"
    "4. Apply the specified less-relevant strategy so the drift increases from index 0 to {last_index}, yet every key "
    "still shares critical terminology or context with the query.\n"
    "5. Never output the original key text or totally irrelevant keys, and avoid meta commentary or disclaimers.\n\n"
    "Return ONLY valid JSON (no Markdown) using:\n"
    "{{\n"
    '  "query": "...",\n'
    '  "query_type": "{query_type}",\n'
    '  "less_relevant_keys": [\n'
    '    "most similar but slightly different key",\n'
    "    ...,\n"
    '    "least similar yet still related key"\n'
    "  ]\n"
    "}}\n\n"
    "Strict rules:\n"
    "- Generate around the same length of keys as original key.\n"
    "- Preserve ordering so keys[0] is more relevant than keys[1], etc., until keys[{last_index}].\n"
    "- If you must clarify assumptions, incorporate them inside the JSON fields.\n"
)


@dataclass(frozen=True)
class Instruction:
    id: str
    query_type: str
    guidance: str


@dataclass(frozen=True)
class LessRelevantInstruction:
    id: str
    guidance: str


@dataclass(frozen=True)
class InstructionPair:
    pair_index: int
    query_instruction: Instruction
    less_instruction: LessRelevantInstruction


QUERY_VARIATIONS: Tuple[Instruction, ...] = (
    Instruction("title_core_concept", "title", "Write a concise 5-7 word title capturing the key's core concept with active wording."),
    Instruction("title_outcome_focus", "title", "Generate a headline-style title that highlights the primary outcome described in the key."),
    Instruction("question_how_process", "question", "Ask a 'How' question that probes the main process or mechanism in the key."),
    Instruction("question_why_motivation", "question", "Ask a 'Why/What causes' question exploring the motivation or rationale behind the key."),
    Instruction("one_sentence_expert", "one_sentence", "Write one dense sentence summarising the key for an expert audience, keeping essential terminology."),
    Instruction("one_sentence_novice", "one_sentence", "Write one clear sentence that explains the key for a newcomer while preserving technical anchors."),
    Instruction("two_sentence_overview", "two_to_three_sentences", "Write two sentences: the first states the core idea, the second adds a critical supporting detail."),
    Instruction("three_sentence_context", "two_to_three_sentences", "Write three short sentences that outline context, action, and expected result from the key."),
    Instruction("paragraph_compact", "paragraph", "Write one compact paragraph (3-4 sentences) that synthesises the key's problem, approach, and impact."),
    Instruction("multi_paragraph_layers", "multi_paragraph", "Write two brief paragraphs: first summarises the main point, second frames implications or comparisons."),
)

LESS_RELEVANT_VARIATIONS: Tuple[LessRelevantInstruction, ...] = (
    LessRelevantInstruction("synonym_substitution", "Gradually replace pivotal entities with near synonyms or closely related alternatives, increasing the shifts each step."),
    LessRelevantInstruction("scope_widening", "Broaden the scenario bit by bit, moving from the exact case toward a more generic context while keeping recognisable ties."),
    LessRelevantInstruction("audience_shift", "Change the intended audience each time (expert, practitioner, executive, student, general public) to soften topical alignment."),
    LessRelevantInstruction("temporal_shift", "Modify the timeframe or recency in stages (current, near future, mid-term, long-term) without leaving the original domain."),
    LessRelevantInstruction("domain_shift", "Move through adjacent domains or industries that still use the core idea, ending with the furthest adjacent but related domain."),
    LessRelevantInstruction("constraint_variation", "Alter a key constraint or parameter per step, making each change more substantial yet still grounded in the same theme."),
    LessRelevantInstruction("problem_fragment", "Focus on increasingly smaller fragments or side aspects of the original problem while keeping vocabulary overlaps."),
    LessRelevantInstruction("tool_swap", "Swap the main tool/technology/process with progressively less direct alternatives that remain plausible for the situation."),
    LessRelevantInstruction("data_shift", "Adjust the data, scale, or metrics stepwise (exact figures, approximate ranges, orders of magnitude) staying contextually relevant."),
    LessRelevantInstruction("perspective_pivot", "Reframe from different stakeholder perspectives, moving from central actor to periphery while maintaining shared concerns."),
)

INSTRUCTION_PAIRS: Tuple[InstructionPair, ...] = tuple(
    InstructionPair(idx, query_inst, less_inst)
    for idx, (query_inst, less_inst) in enumerate(
        (q, l) for q in QUERY_VARIATIONS for l in LESS_RELEVANT_VARIATIONS
    )
)


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


def _extract_terms(text: str, *, limit: int = 6) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)
    seen: List[str] = []
    for token in tokens:
        lower = token.lower()
        if lower not in seen:
            seen.append(lower)
        if len(seen) >= limit:
            break
    return seen or ["topic"]


def _fallback_query(text: str, query_type: str, terms: Sequence[str]) -> str:
    base = " ".join(text.strip().split())
    if not base:
        raise Exception("Error")
        base = "Placeholder content about the topic."
    topic = " ".join(terms[:3]) if terms else "the topic"
    if query_type == "title":
        return " ".join(word.capitalize() for word in terms[:6]) or topic.title()
    if query_type == "question":
        return f"How does {topic} work in practice?"
    if query_type == "one_sentence":
        return f"{topic.capitalize()} overview: {base[:180].strip()}."
    if query_type == "two_to_three_sentences":
        snippet = base[:280].strip()
        if not snippet.endswith("."):
            snippet += "."
        return f"{snippet} What key detail underpins {topic}?"
    if query_type == "paragraph":
        return (
            f"{topic.capitalize()} involves {base[:220].strip()}, explaining the challenge, approach, and expected outcomes in one paragraph."
        )
    if query_type == "multi_paragraph":
        return (
            f"{topic.capitalize()} summary: {base[:160].strip()}. "
            f"In addition, consider broader implications for {terms[-1] if terms else 'related areas'}."
        )
    return base[:200]


def _fallback_less_keys(
    text: str,
    *,
    n_keys: int,
    query: str,
    terms: Sequence[str],
) -> List[str]:
    base = " ".join(text.strip().split())
    topic = terms[0] if terms else "topic"
    secondary = terms[1:] or ["applications", "ecosystem", "workflow", "case study", "alternatives"]
    templates = [
        "Simplified {topic} walkthrough for {detail}.",
        "Key considerations when applying {topic} to {detail}.",
        "Overview of {topic} with emphasis on {detail}.",
        "Practical guide connecting {topic} to {detail}.",
        "Comparative look at {topic} versus {detail}.",
        "Case study: {topic} in {detail}.",
        "Strategic outlook on {topic} for {detail}.",
        "Lessons learned bringing {topic} into {detail}.",
        "Challenges when extending {topic} toward {detail}.",
        "Speculative future of {topic} impacting {detail}.",
    ]
    results: List[str] = []
    for idx in range(n_keys):
        detail = secondary[idx % len(secondary)]
        template = templates[min(idx, len(templates) - 1)]
        candidate = template.format(topic=topic, detail=detail)
        if idx == 0:
            candidate = f"{query} for {detail}"
        elif idx == 1 and base:
            candidate = f"{base[:160].strip()} (focused on {detail})"
        results.append(candidate)
    return results[:n_keys]


def _build_prompt(tokenizer, *, n_keys: int) -> PromptBuilder:
    def build(sentence: str, pair: InstructionPair) -> str:
        content = _USER_TEMPLATE.format(
            sentence=sentence,
            query_instruction_id=pair.query_instruction.id,
            query_instruction=pair.query_instruction.guidance,
            less_instruction_id=pair.less_instruction.id,
            less_instruction=pair.less_instruction.guidance,
            n_keys=n_keys,
            last_index=n_keys - 1,
            query_type=pair.query_instruction.query_type,
        )
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    return build


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _normalise_less_keys(
    values: Any,
    *,
    n_keys: int,
    text: str,
    query: str,
    terms: Sequence[str],
) -> List[str]:
    result: List[str] = []
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                if "key" in item and isinstance(item["key"], str):
                    result.append(item["key"].strip())
                elif "text" in item and isinstance(item["text"], str):
                    result.append(item["text"].strip())
                elif "candidate" in item and isinstance(item["candidate"], str):
                    result.append(item["candidate"].strip())
            elif isinstance(item, str):
                result.append(item.strip())
            elif item is not None:
                result.append(str(item).strip())
    result = [candidate for candidate in result if candidate]
    if len(result) >= n_keys:
        return result[:n_keys]
    raise Exception("Error")
    fallback_needed = n_keys - len(result)
    fallback_values = _fallback_less_keys(text, n_keys=fallback_needed, query=query, terms=terms)
    result.extend(fallback_values)
    return result[:n_keys]


def _generate_stub_records(texts: Sequence[str], *, n_keys: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, text in enumerate(texts):
        terms = _extract_terms(text)
        pair = INSTRUCTION_PAIRS[idx % len(INSTRUCTION_PAIRS)]
        query = _fallback_query(text, pair.query_instruction.query_type, terms)
        less_keys = _fallback_less_keys(text, n_keys=n_keys, query=query, terms=terms)
        records.append(
            {
                "query": query,
                "correspond_key": text,
                "less_relevant_keys": less_keys,
                "df_id": idx,
                "query-type": pair.query_instruction.query_type,
            }
        )
    return records


def _parse_outputs(
    outputs: Iterable[Tuple[int, str]],
    texts: Sequence[str],
    instruction_map: Dict[int, InstructionPair],
    *,
    n_keys: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for request_id, raw_output in outputs:

        original_text = texts[request_id]
        pair = instruction_map[request_id]
        terms = _extract_terms(original_text)
        payload = _extract_json_payload(raw_output) or {}
        query = payload.get("query")
        less_values = payload.get("less_relevant_keys")

        try:
            if not isinstance(query, str) or not query.strip():
                query = _fallback_query(original_text, pair.query_instruction.query_type, terms)
            else:
                query = query.strip()
            query_type = payload.get("query_type")
            if not isinstance(query_type, str) or not query_type.strip():
                query_type = pair.query_instruction.query_type
            
            less_keys = _normalise_less_keys(
                less_values,
                n_keys=n_keys,
                text=original_text,
                query=query,
                terms=terms,
            )
            records.append(
                {
                    "query": query,
                    "correspond_key": original_text,
                    "less_relevant_keys": less_keys,
                    "df_id": request_id,
                    "query-type": query_type,
                }
            )
        except Exception:
            print()
            print("query: ", query)
            print("less_values: ", less_values)
            continue
    return records


def make_query_and_less_relevant_keys_recs(
    texts: Sequence[str],
    *,
    n_key_generation: int,
    tokenizer: Any | None = None,
    request_func: Optional[RequestFunc] = None,
    batch_size: int = 8,
    sampling_config: Optional[Dict[str, Any]] = None,
    timeout_s: Optional[float] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build query and ranked less-relevant key records for each input text."""

    if n_key_generation <= 0:
        raise ValueError("n_key_generation must be a positive integer")

    engine_kwargs = engine_kwargs or {}

    tokenizer_source = engine_kwargs.get("tokenizer_name") or engine_kwargs.get("model_name") if engine_kwargs else None
    if request_func is None and tokenizer is None:
        if not tokenizer_source:
            raise ValueError("Provide tokenizer or include 'model_name' / 'tokenizer_name' in engine_kwargs when request_func is omitted")
        try:
            tokenizer = _load_tokenizer(tokenizer_source)
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            logger.warning("Falling back to stub generation because the tokenizer failed to load: %s", exc)
            return _generate_stub_records(texts, n_keys=n_key_generation)

    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    build_prompt = _build_prompt(tokenizer, n_keys=n_key_generation)
    prompts: List[str] = []
    instruction_map: Dict[int, InstructionPair] = {}
    for idx, sentence in enumerate(texts):
        pair = INSTRUCTION_PAIRS[idx % len(INSTRUCTION_PAIRS)]
        instruction_map[idx] = pair
        prompts.append(build_prompt(sentence, pair))

    if not prompts:
        return []

    print("n requests: ", len(prompts))
    prompts= prompts[:24]

    outputs: List[Tuple[int, str]]
    if request_func is None:
        if engine_kwargs is None:
            raise ValueError("engine_kwargs must be provided when request_func is omitted")

        try:
            from ..utils import vllm_generate_gptoss as _vllm_generate
            from vllm import SamplingParams
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            logger.warning("Falling back to stub generation because vLLM could not be imported: %s", exc)
            return _generate_stub_records(texts, n_keys=n_key_generation)

        engine_params = dict(engine_kwargs)
        model_name = engine_params.pop("model_name", None)
        if not model_name:
            raise ValueError("engine_kwargs must include 'model_name'")

        tensor_parallel_size = engine_params.pop("tensor_parallel_size", 1)
        num_instances = engine_params.pop("num_instances", 1)
        engine_params.pop("tokenizer_name", None)

        sampling_values: Dict[str, Any] = {"temperature": 0.2, "max_tokens": 10000, "top_p": 0.9, "min_tokens": 3000}
        if sampling_config:
            sampling_values.update(sampling_config)
        sampling = SamplingParams(**sampling_values)

        llm = None
        outputs_texts: Optional[List[str]] = None
        try:
            llm = _vllm_generate.build_llm(
                model_name=model_name,
                tensor_parallel_size=tensor_parallel_size,
                num_instances=num_instances,
                **engine_params,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            logger.warning("Falling back to stub generation because the vLLM worker failed to start: %s", exc)
            return _generate_stub_records(texts, n_keys=n_key_generation)

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
            return _generate_stub_records(texts, n_keys=n_key_generation)

        outputs = list(enumerate(outputs_texts))
    else:
        responses = _maybe_run_async(request_func(prompts))
        outputs = list(enumerate(responses))

    return _parse_outputs(outputs, texts, instruction_map, n_keys=n_key_generation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate queries with ordered less-relevant keys for source documents.")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV file containing source keys.")
    parser.add_argument("--text-column", type=str, default="text", help="Column containing the key text.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file for generated records.")
    parser.add_argument("--model-name", type=str, required=True, help="Local vLLM model path or identifier.")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Optional tokenizer name; defaults to --model-name.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of tensor parallel shards per worker.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of worker processes to launch.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="GPU memory utilisation passed to vLLM.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional maximum model context length.")
    parser.add_argument("--dtype", type=str, default=None, help="Optional dtype override for the vLLM engine.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code when loading from HF Hub.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of prompts sent per generation batch.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature used for generation.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value.")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Maximum tokens generated per prompt.")
    parser.add_argument("--timeout-s", type=float, default=None, help="Optional timeout (in seconds) for the overall job.")
    parser.add_argument("--n-key-generation", type=int, default=5, help="Number of less_relevant keys to produce per query.")
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in {args.input_csv}")

    subset = df[df[args.text_column].notna()].copy()
    sentences = subset[args.text_column].astype(str).tolist()
    sentences= sentences[:8]
    print(len(sentences))
    # quit()
    if not sentences:
        raise ValueError("No keys available for generation.")

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

    records = make_query_and_less_relevant_keys_recs(
        sentences,
        n_key_generation=args.n_key_generation,
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
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Saved query and less-relevant key records to {args.output}")
