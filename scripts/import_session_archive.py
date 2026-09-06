#!/usr/bin/env python3
"""Import existing session-corpus text into the private Markdown archive.

The importer deliberately accepts only the generated corpus format, whose
lines contain a source JSONL path and line number. It does not discover,
interpret, or synthesize sessions from arbitrary files. Existing destination
files are never overwritten.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

SOURCE_LINE = re.compile(r"^\[[^\]]+#L\d+\]")


def import_corpus(source_root: Path, destination: Path, manifest_path: Path) -> dict[str, int]:
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    destination.mkdir(parents=True, exist_ok=True)
    imported = skipped = 0
    records = []
    for source in sorted(source_root.glob("*.txt")):
        lines = source.read_text(encoding="utf-8").splitlines()
        if not lines or not all(SOURCE_LINE.match(line) for line in lines if line.strip()):
            raise ValueError(f"source is not a generated session corpus: {source}")
        relative = source.relative_to(source_root).with_suffix(".md")
        target = destination / relative
        if target.exists():
            skipped += 1
            content = target.read_text(encoding="utf-8")
            records.append(
                {
                    "source": str(source),
                    "destination": str(target),
                    "status": "existing",
                    "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "lines": len(content.splitlines()),
                }
            )
            continue
        content = "# Session corpus {}\n\n{}\n".format(source.stem, "\n".join(lines))
        target.write_text(content, encoding="utf-8")
        imported += 1
        records.append(
            {
                "source": str(source),
                "destination": str(target),
                "status": "imported",
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "lines": len(lines),
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"imported": imported, "skipped": skipped}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(import_corpus(args.source_root, args.destination, args.manifest), sort_keys=True)
    )
