import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
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

DEFAULT_MAX_FILE_TOKENS = 4_096
MAX_LINE_LENGTH_IN_PROMPT = 160
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
    # Optional enhancements captured from the LLM output:
    # - difficulties: one label per task (e.g., Easy/Medium/Hard/VeryHard)
    # - answer_notes: short rationale for why these lines are impactful
    difficulties: Optional[List[str]] = None
    answer_notes: Optional[str] = None


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


def tokens_to_char_budget(token_budget: int) -> int:
    if token_budget <= 0:
        return 0
    # Rough heuristic: assume 1 token ~= 4 characters.
    return max(token_budget * 4, 0)


def _dir_context_for_file(code_file: "CodeFile", max_items: int = 40) -> str:
    """
    Build a compact, budget-friendly directory snapshot for the file's folder.
    Shows siblings and subdirs (one level), with language + line counts where possible.
    """
    folder = code_file.absolute_path.parent
    entries: List[str] = []

    try:
        names = sorted(os.listdir(folder))
    except OSError:
        names = []

    count = 0
    for name in names:
        if count >= max_items:
            entries.append("… (truncated)")
            break
        p = folder / name
        try:
            if p.is_dir():
                entries.append(f"[DIR] {p.name}/")
                count += 1
                continue
            # file:
            ext = p.suffix.lower()
            lang = LANGUAGE_BY_EXTENSION.get(ext, ext.lstrip(".") or "text")
            # quick line count without loading too much:
            n_lines = 0
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as fh:
                    for n_lines, _ in enumerate(fh, 1):
                        if n_lines >= 10_000:
                            break
            except OSError:
                n_lines = 0
            marker = " (focus)" if p == code_file.absolute_path else ""
            entries.append(f"     {p.name}  [{lang}]  ~{n_lines} lines{marker}")
            count += 1
        except OSError:
            continue

    return "\n".join(entries) if entries else "(folder not readable)"


def _nearby_file_summaries(code_file: "CodeFile", max_files: int = 10, max_len: int = 90) -> str:
    """
    Provide quick summaries of sibling files: first non-empty, non-whitespace line.
    Keeps it language-agnostic and cheap to compute.
    """
    folder = code_file.absolute_path.parent
    summaries: List[str] = []
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return "(no summaries)"

    for name in names:
        p = folder / name
        if p == code_file.absolute_path or not p.is_file():
            continue
        if len(summaries) >= max_files:
            summaries.append("… (truncated)")
            break
        try:
            first = ""
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    s = line.strip()
                    if s:
                        first = s
                        break
            ext = p.suffix.lower()
            lang = LANGUAGE_BY_EXTENSION.get(ext, ext.lstrip(".") or "text")
            summaries.append(f"- {p.name} [{lang}]: {shorten(first, width=max_len, placeholder='…')}")
        except OSError:
            continue
    return "\n".join(summaries) if summaries else "(no sibling files)"


def build_file_prompt(
    project_name: str,
    code_file: CodeFile,
    max_file_tokens: int,
) -> str:
    char_budget = tokens_to_char_budget(max_file_tokens)
    header_lines = [
        f"You are curating *high-impact* coding evaluation tasks for project '{project_name}'.",
        f"Focus on file '{code_file.relative_path.as_posix()}' ({code_file.language}).",
        "",
        "Context — directory snapshot (current folder):",
        _dir_context_for_file(code_file),
        "",
        "Context — nearby file summaries:",
        _nearby_file_summaries(code_file),
        "",
        "Goal:",
        "- Identify up to 3 *critical* locations where dropping code lines would create challenging, realistic bug-fix or implementation tasks.",
        "- Prefer lines that are pivotal to algorithm correctness, IO boundaries, API contracts, error handling, state management, security, or inter-file dependencies.",
        "- Avoid trivial edits (e.g., cosmetic prints, simple wrappers). Use contiguous ranges where possible.",
        "",
        "Output format (JSON array): For each task object include:",
        '  - "dropped_lines": array of 1-based line numbers (use contiguous ranges where possible),',
        '  - "tasks": 2–3 concise, high-impact prompts written like real tickets,',
        '  - "difficulties": array of labels (one per task) from {"Easy","Medium","Hard","VeryHard"},',
        '  - Optional "file_path" (defaults to this file), and optional "answer_notes" giving a 1–2 sentence rationale for why these lines are critical.',
        "",
        "Respond with JSON only. Do not include prose outside JSON.",
        "",
        "File content:",
    ]

    parts: List[str] = []
    header_text = "\n".join(header_lines) + "\n"
    total_chars = len(header_text)
    unlimited = char_budget == 0
    truncated = False

    for idx, raw_line in enumerate(code_file.lines, start=1):
        sanitized = raw_line.replace("\t", "    ")
        if len(sanitized) > MAX_LINE_LENGTH_IN_PROMPT:
            sanitized = sanitized[: MAX_LINE_LENGTH_IN_PROMPT - 1] + "…"
        formatted = f"{idx:5d}: {sanitized}"
        projected = total_chars + len(formatted) + 1
        if not unlimited and projected > char_budget:
            truncated = True
            break
        parts.append(formatted)
        total_chars = projected

    if truncated:
        parts.append("... (truncated due to prompt budget)")

    return header_text + "\n".join(parts)


def call_planner(
    model,
    project_name: str,
    code_files: List[CodeFile],
    sampling_params: SamplingParams,
    worker_batch_size: int,
    timeout: float,
    max_file_tokens: int,
) -> Dict[str, List[Dict[str, object]]]:
    if not code_files:
        return {}

    prompts: List[str] = []
    rel_paths: List[Path] = []

    for code_file in code_files:
        prompts.append(build_file_prompt(project_name, code_file, max_file_tokens))
        rel_paths.append(code_file.relative_path)

    print("Number of prompts: ", len(prompts))

    responses = generate(
        model=model,
        prompts=prompts,
        batch_size=worker_batch_size,
        timeout_s=timeout,
        sampling_params=sampling_params,
    )

    selections: Dict[str, List[Dict[str, object]]] = {}
    if responses is None:
        responses = []

    for index, rel_file in enumerate(rel_paths):
        response = responses[index] if index < len(responses) else ""
        if not response:
            selections[rel_file.as_posix()] = []
            continue
        start = response.find("[")
        end = response.rfind("]")
        if start == -1 or end == -1:
            selections[rel_file.as_posix()] = []
            continue
        payload = response[start : end + 1]
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            selections[rel_file.as_posix()] = []
            continue
        if isinstance(data, list):
            selections[rel_file.as_posix()] = [
                item for item in data if isinstance(item, dict)
            ]
        else:
            selections[rel_file.as_posix()] = []

    return selections


def placeholder_for_extension(ext: str, override: Optional[str]) -> str:
    # Placeholder is no longer used; keep function to preserve surface area.
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
    """
    Do NOT modify the file content anymore. Only collect and return the removed source
    lines so that the dataset captures the ground truth without inserting placeholders.
    """
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

    # We deliberately do not write back to disk and do not modify the file.
    # newline = "\n" if text.endswith("\n") else ""
    # updated = "\n".join(lines) + newline
    # file_path.write_text(updated, encoding="utf-8")
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
    default_rel_file: Optional[Path] = None,
) -> Optional[TaskSelection]:
    rel_file_value = selection.get("file_path") or selection.get("path")
    rel_file: Optional[Path]
    if rel_file_value:
        rel_file = sanitize_relative_path(str(rel_file_value))
    elif default_rel_file is not None:
        rel_file = default_rel_file
    else:
        rel_file = None
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

    # Difficulties: accept "difficulties" (list) or "difficulty" (str)
    raw_diffs = selection.get("difficulties")
    difficulties: Optional[List[str]] = None
    if isinstance(raw_diffs, list):
        difficulties = [str(x).strip() for x in raw_diffs if str(x).strip()]
    else:
        single = selection.get("difficulty")
        if isinstance(single, str) and single.strip():
            difficulties = [single.strip()] * len(tasks)

    # Ensure alignment if lengths mismatch
    if difficulties and len(difficulties) != len(tasks):
        diffs = (difficulties + ["Unknown"] * len(tasks))[: len(tasks)]
        difficulties = diffs

    answer_notes = None
    raw_notes = selection.get("answer_notes")
    if isinstance(raw_notes, list):
        answer_notes = " ".join([str(x).strip() for x in raw_notes if str(x).strip()]) or None
    elif isinstance(raw_notes, str):
        answer_notes = raw_notes.strip() or None

    return TaskSelection(
        rel_file=rel_file,
        dropped_lines=line_numbers,
        tasks=tasks,
        difficulties=difficulties,
        answer_notes=answer_notes,
    )


def fallback_selection_for_file(
    code_file: CodeFile,
    fallback_lines: int,
) -> Optional[Dict[str, object]]:
    if fallback_lines <= 0:
        fallback_lines = 1

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
        return None

    description = (
        f"Restore the missing code in {code_file.relative_path.as_posix()} "
        f"around lines {candidates[0]}-{candidates[-1]}."
    )
    return {
        "file_path": code_file.relative_path.as_posix(),
        "dropped_lines": candidates,
        "tasks": [
            description,
            f"Fill the removed implementation near line {candidates[0]} "
            f"in {code_file.relative_path.as_posix()}.",
        ],
    }


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
        "removed_source": dropped_content,
        **({"difficulties": selection.difficulties} if selection.difficulties else {}),
        **({"answer_notes": selection.answer_notes} if selection.answer_notes else {}),
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
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--max-file-tokens",
        type=int,
        default=DEFAULT_MAX_FILE_TOKENS,
        help="Approximate token budget per file prompt (<=0 to include entire file).",
    )
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
        help="Optional override placeholder text inserted on dropped lines (unused).",
    )
    args = parser.parse_args(argv)

    projects = discover_projects(args.code_dir)
    if args.max_projects:
        projects = projects[: args.max_projects]

    dataset: List[Dict[str, object]] = []
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    model = None

    try:
        for project in projects:
            code_files = collect_code_files(project)
            if not code_files:
                continue

            if model is None:
                model = build_llm(
                    model_name=args.model_name,
                    tensor_parallel_size=args.tensor_parallel_size,
                    num_instances=args.num_instances,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    dtype=args.dtype,
                    trust_remote_code=args.trust_remote_code,
                )

            selections_by_file = call_planner(
                model=model,
                project_name=project.name,
                code_files=code_files,
                sampling_params=sampling_params,
                worker_batch_size=args.worker_batch_size,
                timeout=args.timeout,
                max_file_tokens=args.max_file_tokens,
            )

            task_index = 0
            for code_file in code_files:
                rel_posix = code_file.relative_path.as_posix()
                raw_selections = selections_by_file.get(rel_posix, [])

                normalized_entries: List[TaskSelection] = []
                for raw_selection in raw_selections:
                    normalized_selection = normalize_selection(
                        raw_selection,
                        args.max_lines_per_task,
                        default_rel_file=code_file.relative_path,
                    )
                    if normalized_selection:
                        normalized_entries.append(normalized_selection)

                if not normalized_entries:
                    fallback_payload = fallback_selection_for_file(
                        code_file, args.fallback_lines
                    )
                    if fallback_payload:
                        fallback_selection = normalize_selection(
                            fallback_payload,
                            args.max_lines_per_task,
                            default_rel_file=code_file.relative_path,
                        )
                        if fallback_selection:
                            normalized_entries.append(fallback_selection)

                for selection in normalized_entries:
                    suffix = make_task_suffix(task_index, selection.rel_file)
                    task_index += 1
                    task_dir = ensure_task_workspace(project, suffix=suffix)
                    target_file = task_dir / selection.rel_file
                    if not target_file.exists():
                        continue
                    dropped_content = replace_code_lines(
                        # placeholder argument retained but ignored
                        target_file, selection.dropped_lines, placeholder=""
                    )
                    if not dropped_content:
                        continue
                    entry = build_dataset_entry(
                        project=project,
                        task_dir=task_dir,
                        selection=selection,
                        dropped_content=dropped_content,
                    )
                    dataset.append(entry)
    finally:
        if model is not None:
            model.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
