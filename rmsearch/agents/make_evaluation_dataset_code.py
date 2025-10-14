import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from vllm import SamplingParams

from rmsearch.utils.vllm_generate import build_llm, generate


EXCLUDED_DIR_NAMES = {
    "__pycache__",
    "__tasks__",
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "node_modules",
    "venv",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    "__pypackages__",
}

HASH_COMMENT_EXTS = {
    ".py",
    ".rb",
    ".r",
    ".sh",
    ".bash",
    ".zsh",
    ".ksh",
    ".pl",
    ".pm",
    ".ps1",
    ".psm1",
    ".psd1",
    ".yaml",
    ".yml",
}

DOUBLE_SLASH_COMMENT_EXTS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".ino",
    ".pde",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".cs",
    ".go",
    ".rs",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".php",
    ".dart",
    ".m",
    ".mm",
    ".zig",
    ".glsl",
    ".vert",
    ".frag",
}

SQL_COMMENT_EXTS = {
    ".sql",
    ".psql",
    ".lua",
}

FORTRAN_COMMENT_EXTS = {
    ".f",
    ".for",
    ".ftn",
    ".f90",
    ".f95",
    ".f03",
    ".f08",
}

BLOCK_COMMENT_EXTS = {
    ".html",
    ".htm",
    ".xml",
    ".vue",
}

C_BLOCK_COMMENT_EXTS = {
    ".css",
    ".scss",
    ".sass",
    ".less",
}

EXTRA_CODE_EXTS = {
    ".bat",
    ".cmake",
    ".toml",
    ".gradle",
    ".groovy",
    ".make",
    ".mk",
    ".proto",
    ".rspec",
}

SUPPORTED_CODE_EXTENSIONS = (
    HASH_COMMENT_EXTS
    | DOUBLE_SLASH_COMMENT_EXTS
    | SQL_COMMENT_EXTS
    | FORTRAN_COMMENT_EXTS
    | BLOCK_COMMENT_EXTS
    | C_BLOCK_COMMENT_EXTS
    | EXTRA_CODE_EXTS
)

LANGUAGE_BY_EXTENSION: Dict[str, str] = {
    ".py": "python",
    ".rb": "ruby",
    ".r": "r",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ksh": "shell",
    ".pl": "perl",
    ".pm": "perl",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "c-header",
    ".hh": "cpp-header",
    ".hpp": "cpp-header",
    ".hxx": "cpp-header",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".php": "php",
    ".dart": "dart",
    ".m": "objective-c",
    ".mm": "objective-c++",
    ".zig": "zig",
    ".glsl": "glsl",
    ".vert": "glsl",
    ".frag": "glsl",
    ".sql": "sql",
    ".psql": "sql",
    ".lua": "lua",
    ".f": "fortran",
    ".for": "fortran",
    ".ftn": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".f08": "fortran",
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".vue": "vue",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".bat": "batch",
    ".cmake": "cmake",
    ".toml": "toml",
    ".gradle": "gradle",
    ".groovy": "groovy",
    ".make": "makefile",
    ".mk": "makefile",
    ".proto": "protobuf",
    ".rspec": "ruby",
}

MAX_PROMPT_CHARS = 12_000
MAX_LINES_PER_FILE_IN_PROMPT = 120
MAX_LINE_LENGTH_IN_PROMPT = 160
MAX_CODE_FILES_FOR_PROMPT = 40
MAX_FILE_BYTES = 200_000


@dataclass
class CodeFile:
    absolute_path: Path
    relative_path: Path
    language: str
    lines: List[str]


@dataclass
class TaskSelection:
    rel_file: Path
    dropped_lines: List[int]
    tasks: List[str]


def discover_projects(root: Path) -> List[Path]:
    projects = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            projects.append(entry)
    return projects


def is_probably_code_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_CODE_EXTENSIONS


def detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    language = LANGUAGE_BY_EXTENSION.get(ext)
    if language:
        return language
    ext_name = ext.lstrip(".")
    return ext_name if ext_name else "text"


def collect_code_files(project_root: Path) -> List[CodeFile]:
    code_files: List[CodeFile] = []
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")
        ]
        for name in filenames:
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            if not is_probably_code_file(path):
                continue
            try:
                if MAX_FILE_BYTES and path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text.strip():
                continue
            code_files.append(
                CodeFile(
                    absolute_path=path,
                    relative_path=path.relative_to(project_root),
                    language=detect_language(path),
                    lines=text.splitlines(),
                )
            )
    code_files.sort(key=lambda item: item.relative_path.as_posix())
    return code_files


def build_project_snapshot(
    project_name: str,
    code_files: List[CodeFile],
    max_chars: int = MAX_PROMPT_CHARS,
) -> str:
    header_lines = [
        f"You are curating coding evaluation tasks for project '{project_name}'.",
        "Identify up to 3 places where removing a few lines would create a meaningful bug-fix or implementation task.",
        'Return a JSON array. Each object must contain: "file_path", "dropped_lines", "tasks".',
        '"file_path": relative path from the project root.',
        '"dropped_lines": array of 1-based line numbers to blank out (prefer contiguous runs, max 12 numbers).',
        '"tasks": array of 2-3 natural language prompts with varied phrasing describing the missing implementation.',
        "Respond with JSON only.",
        "",
        "Project snapshot:",
    ]
    parts: List[str] = []
    total_chars = sum(len(line) + 1 for line in header_lines)
    truncated = False

    for code_file in code_files[:MAX_CODE_FILES_FOR_PROMPT]:
        file_header = f"=== {code_file.relative_path.as_posix()} ({code_file.language}) ==="
        total_chars += len(file_header) + 1
        if total_chars > max_chars:
            truncated = True
            break
        parts.append(file_header)
        for idx, raw_line in enumerate(
            code_file.lines[:MAX_LINES_PER_FILE_IN_PROMPT], start=1
        ):
            sanitized = raw_line.replace("\t", "    ")
            if len(sanitized) > MAX_LINE_LENGTH_IN_PROMPT:
                sanitized = sanitized[: MAX_LINE_LENGTH_IN_PROMPT - 1] + "…"
            formatted = f"{idx:4d}: {sanitized}"
            if total_chars + len(formatted) + 1 > max_chars:
                truncated = True
                break
            parts.append(formatted)
            total_chars += len(formatted) + 1
        if truncated:
            break

    if truncated:
        parts.append("... (truncated snapshot)")

    return "\n".join(header_lines + parts)


def call_planner(
    project_name: str,
    code_files: List[CodeFile],
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
) -> List[Dict[str, object]]:
    if not code_files:
        return []

    prompt = build_project_snapshot(project_name, code_files, MAX_PROMPT_CHARS)
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
        responses = generate(
            model=model,
            prompts=[prompt],
            batch_size=worker_batch_size,
            timeout_s=timeout,
            sampling_params=sampling_params,
        )
    finally:
        model.close()

    if not responses:
        return []

    response = responses[0]
    start = response.find("[")
    end = response.rfind("]")
    if start == -1 or end == -1:
        return []

    payload = response[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, dict)]


def placeholder_for_extension(ext: str, override: Optional[str]) -> str:
    if override:
        return override
    if ext in HASH_COMMENT_EXTS or ext in {".yaml", ".yml"}:
        return "# TODO: implement missing code"
    if ext in SQL_COMMENT_EXTS:
        return "-- TODO: implement missing code"
    if ext in FORTRAN_COMMENT_EXTS:
        return "! TODO: implement missing code"
    if ext in C_BLOCK_COMMENT_EXTS:
        return "/* TODO: implement missing code */"
    if ext in BLOCK_COMMENT_EXTS:
        return "<!-- TODO: implement missing code -->"
    return "// TODO: implement missing code"


def replace_code_lines(
    file_path: Path,
    line_numbers: List[int],
    placeholder: str,
) -> List[Dict[str, object]]:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    lines = text.splitlines()
    replaced: List[Dict[str, object]] = []
    for lineno in line_numbers:
        if 1 <= lineno <= len(lines):
            original = lines[lineno - 1]
            replaced.append({"lineno": lineno, "source": original})
            match = re.match(r"[ \t]*", original)
            indent = match.group(0) if match else ""
            lines[lineno - 1] = f"{indent}{placeholder}" if indent else placeholder

    if not replaced:
        return []

    newline = "\n" if text.endswith("\n") else ""
    updated = "\n".join(lines) + newline
    file_path.write_text(updated, encoding="utf-8")
    return replaced


def sanitize_relative_path(value: str) -> Optional[Path]:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return None
    parts = [part for part in candidate.parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return None
    if not parts:
        return None
    return Path(*parts)


def parse_line_numbers(raw_lines: object, max_lines: int) -> List[int]:
    numbers: List[int] = []
    if raw_lines is None:
        return numbers

    def add_number(value: int) -> None:
        if value > 0:
            numbers.append(value)

    if isinstance(raw_lines, (list, tuple)):
        tokens = list(raw_lines)
    else:
        tokens = [raw_lines]

    for token in tokens:
        if isinstance(token, int):
            add_number(token)
            continue
        if isinstance(token, str):
            for piece in re.split(r"[,\s]+", token.strip()):
                if not piece:
                    continue
                range_match = re.match(r"^(\d+)\s*[-–—]\s*(\d+)$", piece)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2))
                    if start > end:
                        start, end = end, start
                    for value in range(start, end + 1):
                        add_number(value)
                    continue
                try:
                    add_number(int(piece))
                except ValueError:
                    continue

    # Preserve order while removing duplicates
    seen = set()
    ordered = []
    for number in numbers:
        if number not in seen:
            seen.add(number)
            ordered.append(number)

    if max_lines and len(ordered) > max_lines:
        ordered = ordered[:max_lines]

    return ordered


def parse_tasks(raw_tasks: object) -> List[str]:
    if raw_tasks is None:
        return []
    if isinstance(raw_tasks, str):
        task = raw_tasks.strip()
        return [task] if task else []
    if isinstance(raw_tasks, list):
        tasks: List[str] = []
        for entry in raw_tasks:
            task = str(entry).strip()
            if task:
                tasks.append(task)
        return tasks
    return []


def normalize_selection(
    selection: Dict[str, object],
    max_lines_per_task: int,
) -> Optional[TaskSelection]:
    rel_file_value = selection.get("file_path") or selection.get("path")
    rel_file = sanitize_relative_path(str(rel_file_value)) if rel_file_value else None
    if rel_file is None:
        return None

    line_numbers = parse_line_numbers(selection.get("dropped_lines"), max_lines_per_task)
    if not line_numbers:
        # fall back to other possible keys
        line_numbers = parse_line_numbers(selection.get("lines"), max_lines_per_task)
    if not line_numbers:
        return None

    tasks = parse_tasks(selection.get("tasks"))
    if not tasks:
        fallback = f"Restore the missing logic in {rel_file.as_posix()} around lines {line_numbers[0]}-{line_numbers[-1]}."
        tasks = [fallback]

    return TaskSelection(rel_file=rel_file, dropped_lines=line_numbers, tasks=tasks)


def fallback_selections(
    code_files: List[CodeFile],
    fallback_lines: int,
) -> List[Dict[str, object]]:
    if fallback_lines <= 0:
        fallback_lines = 1
    for code_file in code_files:
        candidates: List[int] = []
        for idx, line in enumerate(code_file.lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("#", "//", "/*", "*", "!", "--")):
                continue
            candidates.append(idx)
            if len(candidates) >= fallback_lines:
                break
        if not candidates:
            continue
        description = (
            f"Restore the missing code in {code_file.relative_path.as_posix()} "
            f"around lines {candidates[0]}-{candidates[-1]}."
        )
        return [
            {
                "file_path": code_file.relative_path.as_posix(),
                "dropped_lines": candidates,
                "tasks": [
                    description,
                    f"Fill the removed implementation near line {candidates[0]} "
                    f"in {code_file.relative_path.as_posix()}.",
                ],
            }
        ]
    return []


def ensure_task_workspace(project: Path, suffix: str) -> Path:
    target_root = project / "__tasks__"
    target_root.mkdir(exist_ok=True)
    target_dir = target_root / suffix
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(project, target_dir, ignore=shutil.ignore_patterns("__tasks__"))
    return target_dir


def make_task_suffix(index: int, rel_file: Path) -> str:
    raw = rel_file.as_posix().replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    if not safe:
        safe = f"file_{index}"
    suffix = f"task_{index}_{safe}"
    return suffix[:80]


def build_dataset_entry(
    project: Path,
    task_dir: Path,
    selection: TaskSelection,
    dropped_content: List[Dict[str, object]],
    placeholder: str,
) -> Dict[str, object]:
    rel_task_dir = task_dir.relative_to(project)
    dropped_lines_text = "-".join(str(num) for num in selection.dropped_lines)
    return {
        "task_id": f"{project.name}:{selection.rel_file.as_posix()}:{dropped_lines_text}",
        "code_dir": project.name,
        "dir_path": rel_task_dir.as_posix(),
        "file_path": selection.rel_file.as_posix(),
        "dropped_lines": selection.dropped_lines,
        "tasks": selection.tasks,
        "placeholder": placeholder,
        "removed_source": dropped_content,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create evaluation dataset by masking code lines across projects."
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        required=True,
        help="Directory containing project subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination dataset JSON file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model used to decide which lines to remove.",
    )
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
    parser.add_argument(
        "--max-lines-per-task",
        type=int,
        default=12,
        help="Maximum number of lines to blank out for a single task.",
    )
    parser.add_argument(
        "--fallback-lines",
        type=int,
        default=3,
        help="Number of lines to blank out if the planner fails.",
    )
    parser.add_argument(
        "--placeholder",
        type=str,
        default=None,
        help="Optional override placeholder text inserted on dropped lines.",
    )
    args = parser.parse_args(argv)

    projects = discover_projects(args.code_dir)
    if args.max_projects:
        projects = projects[: args.max_projects]

    dataset: List[Dict[str, object]] = []

    for project in projects:
        code_files = collect_code_files(project)
        if not code_files:
            continue

        selections_raw = call_planner(
            project_name=project.name,
            code_files=code_files,
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

        if not selections_raw:
            selections_raw = fallback_selections(code_files, args.fallback_lines)

        normalized: List[TaskSelection] = []
        for selection in selections_raw:
            normalized_selection = normalize_selection(selection, args.max_lines_per_task)
            if normalized_selection:
                normalized.append(normalized_selection)

        for index, selection in enumerate(normalized):
            suffix = make_task_suffix(index, selection.rel_file)
            task_dir = ensure_task_workspace(project, suffix=suffix)
            target_file = task_dir / selection.rel_file
            if not target_file.exists():
                continue
            placeholder = placeholder_for_extension(
                target_file.suffix.lower(), args.placeholder
            )
            dropped_content = replace_code_lines(
                target_file, selection.dropped_lines, placeholder
            )
            if not dropped_content:
                continue
            entry = build_dataset_entry(
                project=project,
                task_dir=task_dir,
                selection=selection,
                dropped_content=dropped_content,
                placeholder=placeholder,
            )
            dataset.append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
