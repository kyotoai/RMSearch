"""Generate query sequences with gradually decreasing relevance for DPO training."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

PromptBuilder = Callable[[str, Sequence["QueryInstruction"]], str]
RequestFunc = Callable[[List[str]], List[str]] | Callable[[List[str]], Awaitable[List[str]]]

_SYSTEM_PROMPT = (
    "You are a meticulous retrieval data generator. "
    "For each confidential key you receive, you must craft multiple search queries that progressively drift "
    "from very tight relevance toward mild topical distance while staying on-topic. "
    "Follow the format exactly and output strict JSON without commentary."
)

_USER_TEMPLATE = (
    "Original key (keep private, never repeat verbatim):\n"
    "'''\n"
    "{key_text}\n"
    "'''\n\n"
    "You must write EXACTLY {n_queries} queries. Treat query[0] as the most relevant and each subsequent query as slightly "
    "less aligned than the previous one, while still clearly related to the key's core meaning.\n\n"
    "Ordered generation plan:\n"
    "{instruction_block}\n"
    "Strict relevance ordering rules:\n"
    "- query[0] MUST be the closest possible match to the key's deep intent.\n"
    "- For every i > 0, query[i] must remain meaningfully related but measurably broader, noisier, or more tangential than query[i-1].\n"
    "- None of the queries may be irrelevant or off-topic; the drift is gradual and controlled.\n"
    "- Respect the specified query type and writing guidance for each index.\n\n"
    "Return ONLY valid JSON with this exact structure (no Markdown wrapping, no extra keys):\n"
    "{{\n"
    '  "queries": [\n'
    '    "query_0 string",\n'
    '    "query_1 string",\n'
    "    ...,\n"
    '    "query_{last_index} string"\n'
    "  ]\n"
    "}}\n"
)


@dataclass(frozen=True)
class QueryInstruction:
    id: str
    query_type: str
    guidance: str


QUERY_VARIATIONS: Tuple[QueryInstruction, ...] = (
    # Title variations
    QueryInstruction("title_core_concept", "title", "Write a concise 5-7 word title capturing the key's central claim using active voice."),
    QueryInstruction("title_outcome_focus", "title", "Craft a headline that highlights the primary outcome or benefit described by the key."),
    QueryInstruction("title_problem_solution", "title", "Produce a sub-10-word title framing the key's main problem and solution."),
    QueryInstruction("title_action_signal", "title", "Create an action-oriented title signalling the key's core technique or approach."),
    # Question variations
    QueryInstruction("question_how_process", "question", "Ask a 'How' question focusing on the key's operational steps or workflow."),
    QueryInstruction("question_why_motivation", "question", "Write a 'Why' question investigating motivations or mechanisms behind the key."),
    QueryInstruction("question_compare_choice", "question", "Pose a decision question comparing the key with plausible alternative strategies."),
    QueryInstruction("question_troubleshoot", "question", "Formulate a troubleshooting question about challenges addressed by the key."),
    # Single sentence variations
    QueryInstruction("single_sentence_thesis", "single-sentence", "Write one declarative sentence summarising the key's thesis with precise terminology."),
    QueryInstruction("single_sentence_context_outcome", "single-sentence", "Produce a single sentence that links the key's context to its desired outcome."),
    QueryInstruction("single_sentence_actor_result", "single-sentence", "Craft one sentence leading with the main actors and the result they achieve."),
    QueryInstruction("single_sentence_problem_angle", "single-sentence", "Compose a sentence focusing on the problem the key tackles and its unique angle."),
    # Several sentences variations
    QueryInstruction("several_sentences_two_step", "several-sentences", "Write two sentences: first summarises the situation, second adds a supporting insight."),
    QueryInstruction("several_sentences_three_part", "several-sentences", "Provide three concise sentences covering context, action, and impact."),
    QueryInstruction("several_sentences_user_focus", "several-sentences", "Compose two sentences tying the key to its intended users and results."),
    QueryInstruction("several_sentences_next_steps", "several-sentences", "Develop two sentences stating the key then hinting at logical next steps."),
    # Single paragraph variations
    QueryInstruction("single_paragraph_weave", "single-paragraph", "Write one paragraph (3-4 sentences) weaving together challenge, response, and benefit."),
    QueryInstruction("single_paragraph_compact", "single-paragraph", "Produce a compact paragraph emphasising problem framing, method, and outcome."),
    QueryInstruction("single_paragraph_story", "single-paragraph", "Compose a single paragraph narrating the key's before-and-after transformation."),
    QueryInstruction("single_paragraph_example", "single-paragraph", "Draft one paragraph that states the key's core claim and illustrates it with an example."),
    # Several paragraphs variations
    QueryInstruction("several_paragraphs_implications", "several-paragraphs", "Write two short paragraphs: first states the main idea, second explores implications or limitations."),
    QueryInstruction("several_paragraphs_context", "several-paragraphs", "Compose two paragraphs where the first introduces the key and the second ties it to broader context."),
    QueryInstruction("several_paragraphs_contrast", "several-paragraphs", "Provide two paragraphs contrasting current practice with the change proposed in the key."),
    QueryInstruction("several_paragraphs_objective_follow_on", "several-paragraphs", "Deliver two short paragraphs: start with the key's objective, then describe follow-on considerations."),
)

_variations_by_type: Dict[str, List[QueryInstruction]] = {}
for instruction in QUERY_VARIATIONS:
    _variations_by_type.setdefault(instruction.query_type, []).append(instruction)
VARIATIONS_BY_TYPE: Dict[str, Tuple[QueryInstruction, ...]] = {
    query_type: tuple(instructions) for query_type, instructions in _variations_by_type.items()
}


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


def _select_instruction_plan(
    *,
    n_queries: int,
    rng: random.Random,
) -> List[QueryInstruction]:
    """Choose a sequence of query instructions while covering multiple query types."""

    if n_queries <= 0:
        raise ValueError("n_queries must be a positive integer")

    all_instructions = list(QUERY_VARIATIONS)
    rng.shuffle(all_instructions)
    remaining = list(all_instructions)

    query_types = list(VARIATIONS_BY_TYPE.keys())
    rng.shuffle(query_types)

    plan: List[QueryInstruction] = []
    used_instr_ids: set[str] = set()

    # Ensure we visit as many query types as possible before repeating.
    for query_type in query_types:
        if len(plan) >= n_queries:
            break
        candidates = list(VARIATIONS_BY_TYPE[query_type])
        rng.shuffle(candidates)
        for candidate in candidates:
            if candidate.id in used_instr_ids:
                continue
            plan.append(candidate)
            used_instr_ids.add(candidate.id)
            break

    # Fill any remaining slots with shuffled instructions.
    idx = 0
    while len(plan) < n_queries and idx < len(remaining):
        candidate = remaining[idx]
        if candidate.id not in used_instr_ids:
            plan.append(candidate)
            used_instr_ids.add(candidate.id)
        idx += 1

    # If n_queries exceeds number of unique instructions, allow repetition.
    while len(plan) < n_queries:
        plan.append(rng.choice(all_instructions))

    return plan[:n_queries]


def _format_instruction_block(plan: Sequence[QueryInstruction]) -> str:
    lines: List[str] = []
    for idx, instruction in enumerate(plan):
        if idx == 0:
            relevance_note = "This MUST be the most relevant query."
        else:
            relevance_note = f"Make this slightly less aligned than query[{idx - 1}] while staying clearly connected."
        lines.append(
            f"- query[{idx}] — type: {instruction.query_type} (instruction id: {instruction.id})\n"
            f"  Guidance: {instruction.guidance}\n"
            f"  Relevance note: {relevance_note}"
        )
    return "\n".join(lines)


def _prompt_builder(tokenizer, *, n_queries: int) -> PromptBuilder:
    def build_prompt(key_text: str, plan: Sequence[QueryInstruction]) -> str:
        instruction_block = _format_instruction_block(plan)
        content = _USER_TEMPLATE.format(
            key_text=key_text,
            n_queries=n_queries,
            instruction_block=instruction_block,
            last_index=n_queries - 1,
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

    return build_prompt


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


def _parse_outputs(
    outputs: Iterable[Tuple[int, str]],
    texts: Sequence[str],
    plans: Dict[int, Sequence[QueryInstruction]],
    *,
    n_queries: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    n_error = 0

    for request_id, raw_output in outputs:
        try:
            payload = _extract_json_payload(raw_output) or {}
            queries = payload.get("queries")
            if not isinstance(queries, list) or len(queries) != n_queries:
                
                raise ValueError(f"Expected {n_queries} queries for df_id {request_id}, received payload: {payload}")
            cleaned_queries: List[str] = []
            for idx, query in enumerate(queries):
                if not isinstance(query, str):
                    raise ValueError(f"Query at index {idx} for df_id {request_id} is not a string.")
                value = query.strip()
                if not value:
                    raise ValueError(f"Query at index {idx} for df_id {request_id} is empty after stripping.")
                cleaned_queries.append(value)

            instruction_plan = plans[request_id]
            query_types = [instruction.query_type for instruction in instruction_plan]

            records.append(
                {
                    "queries": cleaned_queries,
                    "key": texts[request_id],
                    "df_id": request_id,
                    "query-types": query_types,
                }
            )
        except Exception:
            print()
            print("----- Error ----")
            print(raw_output)
            n_error += 1
            continue

    print(f"Errors: {n_error} / {len(outputs)}")

    return records


def make_query_dpo_pairs(
    texts: Sequence[str],
    *,
    n_query_generation: int,
    tokenizer: Any | None = None,
    request_func: Optional[RequestFunc] = None,
    batch_size: int = 8,
    sampling_config: Optional[Dict[str, Any]] = None,
    timeout_s: Optional[float] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build ranked query sets for each input text to support DPO training."""

    if n_query_generation <= 0:
        raise ValueError("n_query_generation must be a positive integer")

    engine_kwargs = engine_kwargs or {}

    tokenizer_source = engine_kwargs.get("tokenizer_name") or engine_kwargs.get("model_name") if engine_kwargs else None
    if request_func is None and tokenizer is None:
        if not tokenizer_source:
            raise ValueError("Provide tokenizer or include 'model_name' / 'tokenizer_name' in engine_kwargs when request_func is omitted")
        tokenizer = _load_tokenizer(tokenizer_source)

    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    rng = random.Random(random_seed)
    build_prompt = _prompt_builder(tokenizer, n_queries=n_query_generation)

    prompts: List[str] = []
    plans: Dict[int, Sequence[QueryInstruction]] = {}
    for idx, text in enumerate(texts):
        plan = _select_instruction_plan(n_queries=n_query_generation, rng=rng)
        plans[idx] = plan
        prompts.append(build_prompt(text, plan))

    if not prompts:
        return []

    print("n requests: ", len(prompts))

    outputs: List[Tuple[int, str]]
    if request_func is None:
        if engine_kwargs is None:
            raise ValueError("engine_kwargs must be provided when request_func is omitted")

        try:
            from ..utils import vllm_generate_gptoss as _vllm_generate
            from vllm import SamplingParams
        except Exception as exc:  # pragma: no cover - dependency on runtime availability
            raise RuntimeError("vLLM generation is unavailable") from exc

        engine_params = dict(engine_kwargs)
        model_name = engine_params.pop("model_name", None)
        if not model_name:
            raise ValueError("engine_kwargs must include 'model_name'")

        tensor_parallel_size = engine_params.pop("tensor_parallel_size", 1)
        num_instances = engine_params.pop("num_instances", 1)
        engine_params.pop("tokenizer_name", None)

        sampling_values: Dict[str, Any] = {"temperature": 0.2, "max_tokens": 6000, "top_p": 0.9, "min_tokens": 512}
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
                prompts,
                sampling_params=sampling,
                batch_size=batch_size,
                timeout_s=timeout_s,
            )
        finally:
            llm.close(kill=outputs_texts is None)

        if outputs_texts is None:
            raise RuntimeError("vLLM generation returned no outputs")

        outputs = list(enumerate(outputs_texts))
    else:
        responses = _maybe_run_async(request_func(prompts))
        outputs = list(enumerate(responses))

    return _parse_outputs(outputs, texts, plans, n_queries=n_query_generation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate query sequences with controlled relevance drift for source keys.")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV file containing source keys.")
    parser.add_argument("--text-column", type=str, default="text", help="Column containing the key text.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file for generated query records.")
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
    parser.add_argument("--max-tokens", type=int, default=6000, help="Maximum tokens generated per prompt.")
    parser.add_argument("--min-tokens", type=int, default=512, help="Minimum tokens generated per prompt.")
    parser.add_argument("--timeout-s", type=float, default=None, help="Optional timeout (in seconds) for the overall job.")
    parser.add_argument("--n-query-generation", type=int, default=5, help="Number of queries to produce per key.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for instruction shuffling.")
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in {args.input_csv}")

    subset = df[df[args.text_column].notna()].copy()
    keys = subset[args.text_column].astype(str).tolist()
    if not keys:
        raise ValueError("No keys available for query generation.")

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

    sampling_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens,
    }

    records = make_query_dpo_pairs(
        keys,
        n_query_generation=args.n_query_generation,
        tokenizer=None,
        request_func=None,
        batch_size=args.batch_size,
        sampling_config=sampling_config,
        timeout_s=args.timeout_s,
        engine_kwargs=engine_kwargs,
        random_seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Saved query DPO pairs to {args.output}")
