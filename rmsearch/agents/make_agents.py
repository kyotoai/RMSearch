import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from vllm import SamplingParams

from rmsearch.agents.prompt_parser import AgentTemplate, parse_prompt_markdown
from rmsearch.utils.vllm_generate import build_llm, generate


@dataclass
class PromptSpec:
    document_path: Path
    group: str
    template: AgentTemplate
    index: int
    variant: int
    composed_prompt: str


def discover_markdown_files(root: Path) -> List[Path]:
    files = []
    for entry in sorted(root.rglob("*.md")):
        if entry.name.startswith("."):
            continue
        files.append(entry)
    return files


def parse_device_groups(raw: Optional[str]) -> Optional[List[List[int]]]:
    if not raw:
        return None
    device_groups: List[List[int]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        device_groups.append([int(token) for token in chunk.split(",") if token.strip()])
    return device_groups or None


def compose_prompt(shared: Sequence[str], template: AgentTemplate) -> str:
    parts: List[str] = []
    if shared:
        parts.append("\n\n".join(shared).strip())
    if template.shared_instructions:
        parts.append("\n\n".join(template.shared_instructions).strip())
    if template.body:
        parts.append(template.body.strip())
    return "\n\n".join(filter(None, parts))


def build_prompt_specs(
    prompt_dir: Path,
    n_variants: int,
) -> Tuple[List[PromptSpec], List[Dict[str, object]]]:
    specs: List[PromptSpec] = []
    instruction_records: List[Dict[str, object]] = []
    files = discover_markdown_files(prompt_dir)

    for doc_index, file_path in enumerate(files):
        document = parse_prompt_markdown(file_path)
        shared = list(document.shared_instructions)

        for instruction in document.instructions:
            instruction_records.append(
                {
                    "kind": "instruction",
                    "group": document.group,
                    "name": instruction.name,
                    "prompt_template": instruction.body,
                    "metadata": instruction.metadata,
                    "source_file": os.path.relpath(file_path, prompt_dir),
                }
            )

        for agent_idx, template in enumerate(document.agents):
            combined_shared = shared + template.shared_instructions
            composed = compose_prompt(combined_shared, template)
            for variant in range(n_variants):
                specs.append(
                    PromptSpec(
                        document_path=file_path,
                        group=document.group,
                        template=template,
                        index=agent_idx,
                        variant=variant,
                        composed_prompt=composed,
                    )
                )

    return specs, instruction_records


def serialize_spec(spec: PromptSpec, text: str, prompt_dir: Path, agent_id: int) -> Dict[str, object]:
    return {
        "agent_id": agent_id,
        "kind": "agent",
        "group": spec.group,
        "name": spec.template.name,
        "variant_index": spec.variant,
        "prompt_template": spec.template.body,
        "composed_prompt": spec.composed_prompt,
        "llm_output": text,
        "metadata": spec.template.metadata,
        "source_file": os.path.relpath(spec.document_path, prompt_dir),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate agent prompts from markdown templates.")
    parser.add_argument("--prompts-dir", type=Path, required=True, help="Directory with markdown prompt files.")
    parser.add_argument("--n-agents", type=int, default=5, help="Number of variants per agent template.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file for agents.")
    parser.add_argument("--model-name", type=str, required=True, help="Model path or identifier for generation.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs per worker.")
    parser.add_argument("--num-instances", type=int, default=1, help="Number of worker instances.")
    parser.add_argument("--device-groups", type=str, default=None, help='Optional device groups string, e.g. "0,1;2,3".')
    parser.add_argument("--worker-batch-size", type=int, default=4, help="Prompts per batch per worker.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout in seconds for generation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling value.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization target.")
    parser.add_argument("--max-model-len", type=int, default=10_000, help="Maximum model context length.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype passed to vLLM.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code execution when loading model.")
    parser.add_argument("--stop", nargs="*", default=None, help="Optional stop sequences.")
    parser.add_argument("--save-intermediate", action="store_true", help="Write intermediate prompts next to output.")
    args = parser.parse_args(argv)

    prompt_dir = args.prompts_dir
    prompt_dir.mkdir(parents=True, exist_ok=True)

    specs, instruction_records = build_prompt_specs(prompt_dir, args.n_agents)

    prompts: List[str] = [spec.composed_prompt for spec in specs]
    results: List[str] = []

    if prompts:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop=args.stop,
        )

        llm_kwargs = {
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
        }

        device_groups = parse_device_groups(args.device_groups)
        model = build_llm(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            num_instances=args.num_instances,
            device_groups=device_groups,
            **llm_kwargs,
        )

        try:
            results = generate(
                model=model,
                prompts=prompts,
                batch_size=args.worker_batch_size,
                timeout_s=args.timeout,
                sampling_params=sampling_params,
            )
        finally:
            model.close()

    if args.save_intermediate:
        prompt_dump = args.output.with_suffix(".prompts.jsonl")
        with prompt_dump.open("w", encoding="utf-8") as handle:
            for spec in specs:
                handle.write(json.dumps({"prompt": spec.composed_prompt}) + "\n")

    records: List[Dict[str, object]] = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    if prompts and len(results) != len(specs):
        raise RuntimeError(f"Expected {len(specs)} generations, received {len(results)}.")

    records.extend(
        serialize_spec(spec, text, prompt_dir, agent_id=i)
        for i, (spec, text) in enumerate(zip(specs, results))
    )

    next_id = len(records)
    for instruction in instruction_records:
        instruction_record = dict(instruction)
        instruction_record["agent_id"] = next_id
        instruction_record["generated_at"] = timestamp
        records.append(instruction_record)
        next_id += 1

    for record in records:
        record.setdefault("generated_at", timestamp)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
