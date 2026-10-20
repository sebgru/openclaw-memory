#!/usr/bin/env python3
"""Convert chunked ``sessions_history`` JSON pages into private Markdown archive files.

Each input file is one saved page: either a JSON list of messages or an object
with a ``messages`` list and optional ``sessionKey``/``offset`` (the shape
``sessions_history`` returns). Pages are grouped by ``--session-key`` (or the
page's own ``sessionKey``) and de-duplicated, so overlapping pages are safe.
Output is chronological regardless of filename: messages are ordered by
``__openclaw.seq`` when every message has one, else by page ``offset``, then
filename, then position. Pages flagged ``truncated``, ``contentTruncated`` or
``contentRedacted`` are rejected unless ``--allow-lossy`` is given. Existing
destination files and an existing manifest are never overwritten. The script
only writes Markdown and a manifest; indexing remains a separate, operator-run
``POST /archive/index``.
"""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


LOSSY_FLAGS = ("truncated", "contentTruncated", "contentRedacted")


def _load_page(
    path: Path, default_key: str | None, allow_lossy: bool
) -> tuple[str, list[dict], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    key = default_key
    offset = 0
    if isinstance(data, dict):
        lossy = [f for f in LOSSY_FLAGS if data.get(f)]
        if lossy and not allow_lossy:
            raise ValueError(f"lossy page ({', '.join(lossy)}): {path}; use --allow-lossy")
        key = key or data.get("sessionKey")
        raw_offset = data.get("offset")
        offset = raw_offset if isinstance(raw_offset, int) else 0
        data = data.get("messages")
    if not isinstance(data, list) or not all(isinstance(m, dict) for m in data):
        raise ValueError(f"not a sessions_history page: {path}")
    if not key:
        raise ValueError(f"no session key for {path}; pass --session-key")
    return key, data, offset


def _meta(message: dict) -> dict:
    meta = message.get("__openclaw")
    return meta if isinstance(meta, dict) else {}


def _seq(message: dict) -> int | None:
    seq = _meta(message).get("seq")
    return seq if isinstance(seq, int) else None


def _when(message: dict) -> str:
    ts = message.get("timestamp")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(ts) if ts else ""


def _fingerprint(message: dict, body: str) -> str:
    ident = _meta(message).get("id") or message.get("id") or message.get("messageId")
    if ident:
        return f"id:{ident}"
    raw = f"{message.get('role')}|{message.get('timestamp')}|{body}"
    return "h:" + hashlib.sha256(raw.encode()).hexdigest()


def import_pages(
    source_root: Path,
    destination: Path,
    manifest_path: Path,
    session_key: str | None = None,
    allow_lossy: bool = False,
) -> dict[str, int]:
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    sessions: dict[str, list[tuple[tuple, dict]]] = {}
    for page in sorted(source_root.glob("*.json")):
        key, messages, offset = _load_page(page, session_key, allow_lossy)
        entries = sessions.setdefault(key, [])
        for index, message in enumerate(messages):
            entries.append(((offset, page.name, index), message))
    ordered: dict[str, list[dict]] = {}
    for key, entries in sessions.items():
        if all(_seq(m) is not None for _, m in entries):
            entries.sort(key=lambda e: _seq(e[1]) or 0)
        else:
            entries.sort(key=lambda e: e[0])
        ordered[key] = [m for _, m in entries]
    destination.mkdir(parents=True, exist_ok=True)
    imported = skipped = 0
    records = []
    for key, messages in sorted(ordered.items()):
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
            when = f" — {_when(message)}" if _when(message) else ""
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
    parser.add_argument("--allow-lossy", action="store_true")
    args = parser.parse_args()
    result = import_pages(
        args.source_root, args.destination, args.manifest, args.session_key, args.allow_lossy
    )
    print(json.dumps(result, sort_keys=True))
