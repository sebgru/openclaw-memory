"""Human-controlled promotion of review candidates into durable knowledge."""

import json
import os
import tempfile
from pathlib import Path


class PromotionError(ValueError):
    pass


def candidates(root: str | Path) -> list[dict[str, str]]:
    root = Path(root).resolve()
    queue = root / "memory" / "review-candidates"
    if not queue.is_dir():
        return []
    return [
        {"path": p.relative_to(root).as_posix(), "modified": str(p.stat().st_mtime_ns)}
        for p in sorted(queue.rglob("*.md"))
        if p.is_file() and not p.is_symlink()
    ]


def promote(root: str | Path, candidate: str, destination: str, approved_by: str) -> dict[str, str]:
    """Promote one candidate after an explicit human approval.

    The candidate must be in the review queue and destination must be in
    memory/knowledge. The write is atomic; the candidate is retained as an
    audit record.
    """
    if not approved_by.strip():
        raise PromotionError("approved_by is required")
    root = Path(root).resolve()
    source = (root / candidate).resolve()
    target = (root / destination).resolve()
    queue = (root / "memory" / "review-candidates").resolve()
    knowledge = (root / "memory" / "knowledge").resolve()
    if source.parent != queue and queue not in source.parents:
        raise PromotionError("candidate must be under memory/review-candidates")
    if knowledge not in target.parents or target.suffix.lower() != ".md":
        raise PromotionError("destination must be a Markdown file under memory/knowledge")
    if not source.is_file() or source.is_symlink():
        raise PromotionError("candidate does not exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_text(encoding="utf-8")
    marker = f"\n\n<!-- promoted from {candidate}; approved by {approved_by.strip()} -->\n"
    fd, temporary = tempfile.mkstemp(prefix=".promotion-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content.rstrip() + marker)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"candidate": candidate, "destination": destination, "approved_by": approved_by.strip()}


def audit_record(result: dict[str, str]) -> str:
    return json.dumps(result, sort_keys=True)
