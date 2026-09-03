"""Small, dependency-free Markdown scanner/chunker."""
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class MarkdownChunk:
    heading: str
    text: str
    start_line: int

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

def chunk_markdown(markdown: str, max_chars: int = 1600) -> list[MarkdownChunk]:
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    sections: list[tuple[str, int, list[str]]] = []
    heading = ""
    start = 1
    lines: list[str] = []
    for number, line in enumerate(markdown.splitlines(), 1):
        match = _HEADING.match(line)
        if match:
            if any(x.strip() for x in lines): sections.append((heading, start, lines))
            heading, start, lines = match.group(2), number, []
        else:
            if not lines: start = number
            lines.append(line)
    if any(x.strip() for x in lines): sections.append((heading, start, lines))
    chunks: list[MarkdownChunk] = []
    for title, line, body in sections:
        paragraphs = re.split(r"\n\s*\n", "\n".join(body).strip())
        current = ""
        current_line = line
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph: continue
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if current and len(candidate) > max_chars:
                chunks.append(MarkdownChunk(title, current, current_line)); current, current_line = paragraph, line
            elif len(candidate) <= max_chars:
                current = candidate
            else:
                for offset in range(0, len(paragraph), max_chars):
                    part = paragraph[offset:offset + max_chars]
                    chunks.append(MarkdownChunk(title, part, line + offset // 80))
                current = ""
        if current: chunks.append(MarkdownChunk(title, current, current_line))
    return chunks
