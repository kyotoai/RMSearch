import argparse
import ast
import json
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from vllm import SamplingParams

from rmsearch.utils.vllm_generate import build_llm, generate


@dataclass
class FunctionSpec:
    file_path: Path
    name: str
    lineno: int
    end_lineno: int
    source: str
    signature: str
    docstring: Optional[str]


def discover_projects(root: Path) -> List[Path]:
    projects = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            projects.append(entry)
    return projects


def load_python_functions(path: Path) -> List[FunctionSpec]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    functions: List[FunctionSpec] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            end_lineno = getattr(node, "end_lineno", node.lineno)
            lines = text.splitlines()
            snippet = "\n".join(lines[node.lineno - 1:end_lineno])
            signature = f"def {node.name}({', '.join(arg.arg for arg in node.args.args)}):"
            docstring = ast.get_docstring(node)
            functions.append(
                FunctionSpec(
                    file_path=path,
                    name=node.name,
                    lineno=node.lineno,
                    end_lineno=end_lineno,
                    source=snippet,
                    signature=signature,
                    docstring=docstring,
                )
            )
    return functions


def summarize_project(project_root: Path) -> Dict[str, List[FunctionSpec]]:
    summary: Dict[str, List[FunctionSpec]] = {}
    for file_path in project_root.rglob("*.py"):
        summary[str(file_path.relative_to(project_root))] = load_python_functions(file_path)
    return summary


def build_planning_prompt(project_name: str, summary: Dict[str, List[FunctionSpec]], max_chars: int) -> str:
    lines = [f"You are curating coding evaluation tasks for project '{project_name}'."]
    lines.append("Select up to 3 functions that can be safely removed to create a meaningful task.")
    lines.append("Return a JSON array with objects: file_path, function, question, answer_notes.")
    lines.append("")
    total = 0
    for rel_path, functions in summary.items():
        lines.append(f"- File: {rel_path}")
        for func in functions:
            entry = f"  - {func.signature}  # lines {func.lineno}-{func.end_lineno}"
            if func.docstring:
                entry += f" | docstring: {func.docstring.strip()}"
            total += len(entry)
            if total > max_chars:
                lines.append("  - ...")
                break
            lines.append(entry)
        if total > max_chars:
            break
    lines.append("\nRespond with JSON only.")
    return "\n".join(lines)


def call_planner(
    project_root: Path,
    project_name: str,
    model_name: str,
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
) -> List[Dict[str, str]]:
    summary = summarize_project(project_root)
    prompt = build_planning_prompt(project_name, summary, max_chars=12_000)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    model = build_llm(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_instances=num_instances,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    try:
        [response] = generate(
            model=model,
            prompts=[prompt],
            batch_size=worker_batch_size,
            timeout_s=timeout,
            sampling_params=sampling_params,
        )
    finally:
        model.close()

    start = response.find("[")
    end = response.rfind("]")
    if start == -1 or end == -1:
        return []
    json_payload = response[start:end + 1]
    try:
        data = json.loads(json_payload)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def ensure_task_workspace(project: Path, suffix: str) -> Path:
    target_root = project / "__tasks__"
    target_root.mkdir(exist_ok=True)
    target_dir = target_root / suffix
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(project, target_dir, ignore=shutil.ignore_patterns("__tasks__"))
    return target_dir


def replace_function_body(file_path: Path, function_name: str, placeholder: str) -> Optional[str]:
    text = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    lines = text.splitlines()
    replacement: Optional[Tuple[int, int]] = None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            start = node.body[0].lineno - 1 if node.body else node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            replacement = (start, end)
            break

    if not replacement:
        return None

    start, end = replacement
    indent = " " * (len(lines[start]) - len(lines[start].lstrip()))
    placeholder_block = textwrap.dedent(placeholder).strip("\n").splitlines()
    placeholder_lines = [f"{indent}{line}" if line.strip() else line for line in placeholder_block]
    new_lines = lines[:start] + placeholder_lines + lines[end:]
    file_path.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return "\n".join(lines[start:end])


def default_question(rel_path: str, function_name: str, description: Optional[str]) -> str:
    prompt = f"Implement the function `{function_name}` in `{rel_path}` so that it matches the original behaviour."
    if description:
        prompt += f"\n\nNotes: {description.strip()}"
    return prompt


def build_dataset_entry(
    project: Path,
    task_dir: Path,
    rel_file: Path,
    function_name: str,
    question: str,
    original_source: str,
    answer_notes: Optional[str],
) -> Dict[str, object]:
    rel_dir = task_dir.relative_to(project)
    return {
        "task_id": f"{project.name}:{rel_dir}:{function_name}",
        "task": question,
        "dir_path": str(rel_dir),
        "file_path": str(rel_file),
        "function": function_name,
        "correct_answer": {
            "source": original_source,
            "answer_notes": answer_notes,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Create evaluation dataset by masking functions in code projects.")
    parser.add_argument("--code-dir", type=Path, required=True, help="Directory containing project subdirectories.")
    parser.add_argument("--output", type=Path, required=True, help="Destination dataset JSON file.")
    parser.add_argument("--model-name", type=str, required=True, help="Model used to plan function removals.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-instances", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8_192)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--worker-batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-projects", type=int, default=None)
    parser.add_argument("--fallback-per-file", type=int, default=1, help="Fallback number of functions per file.")
    parser.add_argument("--placeholder", type=str, default="raise NotImplementedError(\"TODO: implement\")")
    args = parser.parse_args(argv)

    projects = discover_projects(args.code_dir)
    if args.max_projects:
        projects = projects[: args.max_projects]

    dataset: List[Dict[str, object]] = []

    for project in projects:
        selections = call_planner(
            project_root=project,
            project_name=project.name,
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            num_instances=args.num_instances,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            worker_batch_size=args.worker_batch_size,
            timeout=args.timeout,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        if not selections:
            summary = summarize_project(project)
            for rel_file, functions in summary.items():
                for func in functions[: args.fallback_per_file]:
                    selections.append(
                        {
                            "file_path": rel_file,
                            "function": func.name,
                            "question": default_question(rel_file, func.name, func.docstring),
                            "answer_notes": func.docstring,
                        }
                    )

        for index, selection in enumerate(selections):
            rel_file = Path(selection["file_path"])
            function_name = selection["function"]
            question = selection.get("question") or default_question(
                str(rel_file), function_name, selection.get("answer_notes")
            )
            answer_notes = selection.get("answer_notes")

            task_dir = ensure_task_workspace(project, suffix=f"task_{index}_{function_name}")
            target_file = task_dir / rel_file
            original = replace_function_body(target_file, function_name, args.placeholder)
            if not original:
                continue

            entry = build_dataset_entry(
                project=project,
                task_dir=task_dir,
                rel_file=rel_file,
                function_name=function_name,
                question=question,
                original_source=original,
                answer_notes=answer_notes,
            )
            dataset.append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
