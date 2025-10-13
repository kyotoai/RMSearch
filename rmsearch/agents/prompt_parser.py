import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Section:
    """Generic section extracted from a markdown file."""
    name: str
    body: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentTemplate(Section):
    kind: str = "agent"
    shared_instructions: List[str] = field(default_factory=list)


@dataclass
class InstructionTemplate(Section):
    kind: str = "instruction"


@dataclass
class PromptDocument:
    path: Path
    group: str
    agents: List[AgentTemplate] = field(default_factory=list)
    instructions: List[InstructionTemplate] = field(default_factory=list)
    shared_instructions: List[str] = field(default_factory=list)


_SECTION_PATTERN = re.compile(r"^##\s*([A-Za-z][\w\s-]*)(?::\s*(.*))?$")
_SUBSECTION_PATTERN = re.compile(r"^###\s*([A-Za-z][\w\s-]*)")


def _split_subsections(body: str) -> Dict[str, str]:
    """Split a block of text into subsections keyed by the ### heading."""
    lines = body.splitlines()
    sections: Dict[str, List[str]] = {}
    current_key = "__root__"
    sections[current_key] = []

    for line in lines:
        match = _SUBSECTION_PATTERN.match(line.strip())
        if match:
            current_key = match.group(1).strip().lower()
            sections[current_key] = []
        else:
            sections[current_key].append(line)

    return {key: "\n".join(value).strip() for key, value in sections.items() if value}


def _extract_metadata(section_map: Dict[str, str]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for key, value in list(section_map.items()):
        if key in {"metadata", "meta"}:
            for raw_line in value.splitlines():
                if ":" not in raw_line:
                    continue
                k, v = raw_line.split(":", 1)
                metadata[k.strip()] = v.strip()
            section_map.pop(key, None)
    return metadata


def parse_prompt_markdown(path: Path) -> PromptDocument:
    """Parse a markdown prompt file into agent and instruction templates."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    group = path.stem

    if lines:
        first_line = lines[0].strip()
        if first_line.startswith("#"):
            group = first_line.lstrip("#").strip() or group

    document = PromptDocument(path=path, group=group)

    current_kind: Optional[str] = None
    current_name: Optional[str] = None
    buffer: List[str] = []

    def flush_section() -> None:
        nonlocal buffer, current_kind, current_name
        if not buffer or current_kind is None:
            buffer = []
            current_kind = None
            current_name = None
            return

        content = textwrap.dedent("\n".join(buffer)).strip()
        buffer = []

        if not content:
            current_kind = None
            current_name = None
            return

        section_map = _split_subsections(content)
        metadata = _extract_metadata(section_map)
        body = section_map.pop("prompt", None) or section_map.pop("__root__", content)

        if current_kind == "agent":
            template = AgentTemplate(
                name=current_name or f"Agent {len(document.agents) + 1}",
                body=body.strip(),
                metadata=metadata,
            )
            template.metadata.update(
                {k: v for k, v in section_map.items() if k not in {"instructions"}}
            )
            extra_instructions = section_map.get("instructions")
            if extra_instructions:
                template.shared_instructions.append(extra_instructions.strip())
            document.agents.append(template)

        elif current_kind == "instruction":
            document.instructions.append(
                InstructionTemplate(
                    name=current_name or f"Instruction {len(document.instructions) + 1}",
                    body=body.strip(),
                    metadata=metadata,
                )
            )

        elif current_kind == "shared":
            document.shared_instructions.append(body.strip())

        current_kind = None
        current_name = None

    for raw_line in lines:
        line = raw_line.rstrip()
        match = _SECTION_PATTERN.match(line.strip())
        if match:
            flush_section()
            heading = match.group(1).strip().lower()
            name = (match.group(2) or "").strip()

            if heading.startswith("agent"):
                current_kind = "agent"
            elif heading.startswith("instruction"):
                current_kind = "instruction"
            elif heading.startswith("shared"):
                current_kind = "shared"
            else:
                # treat unknown headings as shared context
                current_kind = "shared"
                if not name:
                    name = match.group(1).strip()

            current_name = name or None
            buffer = []
            continue

        buffer.append(line)

    flush_section()

    return document

