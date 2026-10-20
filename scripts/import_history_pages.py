#!/usr/bin/env python3
"""Convert chunked ``sessions_history`` JSON pages into private Markdown archive files.

Each input file is one saved page: either a JSON list of messages or an object
with a ``messages`` list and optional ``sessionKey``. Pages are grouped by
``--session-key`` (or the page's own ``sessionKey``), concatenated in filename
order, and de-duplicated so overlapping pages are safe. Existing destination
files are never overwritten. The script only writes Markdown and a manifest;
indexing remains a separate, operator-run ``POST /archive/index``.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _load_page(path: Path, default_key: str | None) -> tuple[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    key = default_key
    if isinstance(data, dict):
        key = key or data.get("sessionKey")
        data = data.get("messages")
    if not isinstance(data, list) or not all(isinstance(m, dict) for m in data):
        raise ValueError(f"not a sessions_history page: {path}")
    if not key:
        raise ValueError(f"no session key for {path}; pass --session-key")
    return key, data


def _fingerprint(message: dict, body: str) -> str:
    ident = message.get("id") or message.get("messageId")
    if ident:
        return f"id:{ident}"
    raw = f"{message.get('role')}|{message.get('timestamp')}|{body}"
    return "h:" + hashlib.sha256(raw.encode()).hexdigest()


def import_pages(
    source_root: Path, destination: Path, manifest_path: Path, session_key: str | None = None
) -> dict[str, int]:
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    sessions: dict[str, list[dict]] = {}
    for page in sorted(source_root.glob("*.json")):
        key, messages = _load_page(page, session_key)
        sessions.setdefault(key, []).extend(messages)
    destination.mkdir(parents=True, exist_ok=True)
    imported = skipped = 0
    records = []
    for key, messages in sorted(sessions.items()):
        seen: set[str] = set()
        lines = []
        for message in messages:
            body = _text(message.get("content")).strip()
            if not body:
                continue
            fp = _fingerprint(message, body)
            if fp in seen:
                continue
            seen.add(fp)
            who = message.get("role") or "unknown"
            when = f" — {message['timestamp']}" if message.get("timestamp") else ""
            lines.append(f"## {who}{when}\n\n{body}\n")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_")[:120] or "session"
        slug += "-" + hashlib.sha256(key.encode()).hexdigest()[:8]
        target = destination / f"{slug}.md"
        record = {"sessionKey": key, "destination": str(target), "messages": len(lines)}
        if target.exists():
            skipped += 1
            record["status"] = "existing"
            record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        else:
            content = f"# Session history {key}\n\n" + "\n".join(lines)
            target.write_text(content, encoding="utf-8")
            imported += 1
            record["status"] = "imported"
            record["sha256"] = hashlib.sha256(content.encode()).hexdigest()
        records.append(record)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"imported": imported, "skipped": skipped}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--session-key")
    args = parser.parse_args()
    print(
        json.dumps(
            import_pages(args.source_root, args.destination, args.manifest, args.session_key),
            sort_keys=True,
        )
    )
