import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from vllm import SamplingParams

from rmsearch.utils.vllm_generate import build_llm, generate


@dataclass
class TaskSample:
    task_id: str
    question: str
    dir_path: Path
    file_path: Path
    correct_answer: Dict[str, object]
    raw_entry: Dict[str, object]


def load_dataset(path: Path) -> List[TaskSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: List[TaskSample] = []
    for entry in data:
        samples.append(
            TaskSample(
                task_id=entry["task_id"],
                question=entry["task"],
                dir_path=Path(entry["dir_path"]),
                file_path=Path(entry["file_path"]),
                correct_answer=entry.get("correct_answer", {}),
                raw_entry=entry,
            )
        )
    return samples


def load_context(sample: TaskSample, code_root: Path, max_chars: int) -> str:
    target_dir = code_root / sample.dir_path
    target_file = target_dir / sample.file_path
    if not target_file.exists():
        return ""
    text = target_file.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n...\n" + tail


def compose_generation_prompt(sample: TaskSample, context: str) -> str:
    header = textwrap.dedent(
        f"""
        You are an autonomous software agent tasked with implementing missing functionality.
        Task ID: {sample.task_id}
        Target file: {sample.file_path}
        """
    ).strip()

    instruction = textwrap.dedent(
        """
        Produce the complete function implementation. Respond using Markdown fenced code blocks when returning code.
        """
    ).strip()

    question = sample.question.strip()

    prompt_parts = [header, question]
    if context:
        prompt_parts.append("Current file contents:\n```\n" + context.strip() + "\n```")
    prompt_parts.append(instruction)
    return "\n\n".join(prompt_parts)


def compose_judge_prompt(question: str, candidate: str, reference: Dict[str, object]) -> str:
    reference_source = reference.get("source", "")
    notes = reference.get("answer_notes", "")
    payload = textwrap.dedent(
        f"""
        Evaluate the candidate solution to the coding task. Respond with a JSON object containing
        keys: correct (boolean), score (0-1 float), explanation (string).

        Task description:
        {question.strip()}

        Reference solution:
        ```
        {reference_source.strip()}
        ```
        Additional notes: {notes}

        Candidate solution:
        ```
        {candidate.strip()}
        ```
        """
    ).strip()
    return payload


def evaluate_with_judge(
    judge_model_name: str,
    prompts: List[str],
    tensor_parallel_size: int,
    num_instances: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    trust_remote_code: bool,
    worker_batch_size: int,
    timeout: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> List[Dict[str, object]]:
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    model = build_llm(
        model_name=judge_model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    try:
        responses = generate(
            model=model,
            prompts=prompts,
            batch_size=worker_batch_size,
            timeout_s=timeout,
            sampling_params=sampling_params,
        )
    finally:
        model.close()

    judgments: List[Dict[str, object]] = []
    for response in responses:
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1:
            judgments.append({"correct": False, "score": 0.0, "explanation": response.strip()})
            continue
        payload = response[start : end + 1]
        try:
            judgments.append(json.loads(payload))
        except json.JSONDecodeError:
            judgments.append({"correct": False, "score": 0.0, "explanation": response.strip()})
    return judgments


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run LLM inference for evaluation dataset.")
    parser.add_argument("--code-dir", type=Path, required=True, help="Directory with prepared code tasks.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset JSON file.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for per-task logs.")
    parser.add_argument("--inference-out", type=Path, required=True, help="Result JSON file.")
    parser.add_argument("--model-name", type=str, required=True, help="Model used for code generation.")
    parser.add_argument("--judge-model-name", type=str, default=None, help="Optional model used for automatic judging.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-instances", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=10_000)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--worker-batch-size", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--context-max-chars", type=int, default=6000)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=0.9)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    samples = load_dataset(args.dataset)
    prompts: List[str] = []
    prompt_index: List[int] = []

    for idx, sample in enumerate(samples):
        context = load_context(sample, args.code_dir, args.context_max_chars)
        prompt = compose_generation_prompt(sample, context)
        prompts.append(prompt)
        prompt_index.append(idx)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    model = build_llm(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        num_instances=args.num_instances,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    try:
        generations = generate(
            model=model,
            prompts=prompts,
            batch_size=args.worker_batch_size,
            timeout_s=args.timeout,
            sampling_params=sampling_params,
        )
    finally:
        model.close()

    if len(generations) != len(prompts):
        raise RuntimeError(f"Expected {len(prompts)} generations, received {len(generations)}.")

    judge_prompts: List[str] = []
    judge_indices: List[int] = []
    results: List[Dict[str, object]] = []
    args.output.mkdir(parents=True, exist_ok=True)

    for idx, output_text in zip(prompt_index, generations):
        sample = samples[idx]
        log_path = args.output / f"{sample.task_id.replace(':', '_')}.json"
        log_payload = {
            "task_id": sample.task_id,
            "question": sample.question,
            "prompt": prompts[idx],
            "output": output_text,
        }
        log_path.write_text(json.dumps(log_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        correctness = {"correct": None, "score": None, "explanation": None}

        reference = sample.correct_answer
        if reference:
            judge_prompts.append(compose_judge_prompt(sample.question, output_text, reference))
            judge_indices.append(len(results))

        results.append(
            {
                "task_id": sample.task_id,
                "question": sample.question,
                "dir_path": str(sample.dir_path),
                "inference_log": str(log_path.relative_to(args.output.parent)),
                "output": output_text,
                "correct_answer": reference,
                "correctness": correctness,
            }
        )

    if judge_prompts:
        judgments = evaluate_with_judge(
            judge_model_name=args.judge_model_name or args.model_name,
            prompts=judge_prompts,
            tensor_parallel_size=args.tensor_parallel_size,
            num_instances=args.num_instances,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            worker_batch_size=1,
            timeout=args.timeout,
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
            max_tokens=args.judge_max_tokens,
        )
        for idx, judgment in zip(judge_indices, judgments):
            results[idx]["correctness"] = {
                "correct": judgment.get("correct"),
                "score": judgment.get("score"),
                "explanation": judgment.get("explanation"),
            }

    args.inference_out.parent.mkdir(parents=True, exist_ok=True)
    args.inference_out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
