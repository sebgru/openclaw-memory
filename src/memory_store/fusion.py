"""Deterministic rank-only fusion and conservative query-time canonicalization.

No query/candidate text or paths are logged. Metadata is optional, never inferred
from file mtime. This is not a temporal fact/conflict resolver.
"""

import hashlib
import math
import re
from datetime import date
from pathlib import PurePosixPath
from typing import Any

RRF_K = 60
COPY_COMPONENT = re.compile(
    r"^(?:startup[-_])?(?:memory[-_])?(?:backup|trim|copies)(?:[-_].*)?$", re.I
)
COPY_SUFFIX = re.compile(
    r"(?:[._-](?:bak|backup|copy|trim)(?:[-_]\d[\d-]*)?|\s*\(copy(?: \d+)?\))$", re.I
)


def reciprocal_rank(rank, weight=1.0):
    """One-based rank contribution; scores from different engines never mix."""
    return weight / (RRF_K + rank)


def canonical_path(path):
    """Remove known copy containers/suffixes, not arbitrary dated directories."""
    parts = str(path or "").replace("\\", "/").casefold().split("/")
    copied = any(COPY_COMPONENT.fullmatch(part) for part in parts)
    parts = [part for part in parts if part not in {"", "."} and not COPY_COMPONENT.fullmatch(part)]
    name = parts[-1] if parts else ""
    # Both report.md.bak and report-copy.md normalize to report.md.
    while name.lower().endswith((".bak", ".backup")):
        name = name.rsplit(".", 1)[0]
        copied = True
    suffix = PurePosixPath(name).suffix
    stem = name[: -len(suffix)] if suffix else name
    cleaned = COPY_SUFFIX.sub("", stem)
    copied = copied or cleaned != stem
    if parts:
        parts[-1] = cleaned + suffix
    return "/".join(parts), copied


def _metadata(row, field):
    metadata = row.get("metadata")
    return row.get(field, metadata.get(field) if isinstance(metadata, dict) else None)


def document_date(row):
    value = _metadata(row, "doc_date")
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def session_date(row):
    """Use explicit dates, then ISO dates in existing transcript headings/paths."""
    observed = document_date(row)
    if observed:
        return observed
    for value in (_metadata(row, "session_date"), row.get("heading"), row.get("path")):
        match = re.search(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", str(value or ""))
        if match:
            try:
                return date.fromisoformat(match[0])
            except ValueError:
                continue
    return None


def tier_prior(row, today=None):
    """Bounded rank-level prior; outputs/projects retain the main-memory lane.

    Main memory has a higher prior than archive; recent dated sessions get T1.
    Unknown/old archive stays T2. Document dates never imply curated authority.
    """
    today = today or date.today()
    source = row.get("source", "memory")
    observed = session_date(row)
    if source in {"memory", "artifact"}:
        return 4.0
    if source == "session":
        return 3.0
    if source == "archive":
        return 3.0 if observed and 0 <= (today - observed).days <= 30 else 2.0
    return 1.0


def metadata_adjustment(row, today=None):
    """Optional, bounded date bonus and OCR penalty; missing metadata is neutral."""
    adjustment = 0.0
    if row.get("source") == "document":
        observed = document_date(row)
        if observed:
            age = max(0, ((today or date.today()) - observed).days)
            adjustment += 0.25 / (1 + age / 365)
        quality = _metadata(row, "ocr_quality")
        if (
            isinstance(quality, (int, float))
            and not isinstance(quality, bool)
            and math.isfinite(quality)
        ):
            adjustment -= 0.25 * (1 - max(0.0, min(1.0, quality)))
    return adjustment


def _identities(row):
    text = " ".join(str(row.get("text", "")).casefold().split())
    content_hash = _metadata(row, "content_hash")
    keys: list[tuple] = []
    if content_hash:
        keys.append(("hash", str(content_hash)))
    if text:
        keys.append(("text", hashlib.sha256(text.encode()).hexdigest()))
    path, _ = canonical_path(row.get("path"))
    if path and (row.get("heading") is not None or row.get("line") is not None):
        # Never collapse different chunks of the same document solely by path.
        keys.append(
            (
                "path",
                "documents" if row.get("source") == "document" else "workspace",
                path,
                str(row.get("heading", "")).casefold(),
                str(row.get("line", "")),
            )
        )
    return keys


def fuse_scopes(lanes, today=None):
    """RRF scope lists, collapse duplicate components, prefer canonical content.

    Duplicate ranks use max, not sum: copies are not independent evidence.
    Distinct scope evidence for one canonical result contributes once per scope.
    """
    groups: list[list[tuple[str, int, dict[str, Any]]]] = []
    index: dict[tuple, int] = {}
    for scope, rows in lanes:
        for rank, row in enumerate(rows, 1):
            keys = _identities(row)
            matches = {index[key] for key in keys if key in index}
            group_id = min(matches) if matches else len(groups)
            if not matches:
                groups.append([])
            for other in matches - {group_id}:
                groups[group_id].extend(groups[other])
                groups[other] = []
                for key, owner in list(index.items()):
                    if owner == other:
                        index[key] = group_id
            groups[group_id].append((scope, rank, row))
            for key in keys:
                index[key] = group_id
    results = []
    for group in groups:
        if not group:
            continue
        # A canonical path always wins over a copy, regardless of hit score.
        _, _, canonical = min(
            group,
            key=lambda entry: (
                canonical_path(entry[2].get("path"))[1],
                bool(entry[2].get("canonical_of")),
                -tier_prior(entry[2], today),
                entry[1],
                str(entry[2].get("id", "")),
            ),
        )
        item = {**canonical}
        per_scope: dict[str, int] = {}
        for scope, rank, _ in group:
            per_scope[scope] = min(rank, per_scope.get(scope, rank))
        base = sum(reciprocal_rank(rank) for rank in per_scope.values())
        copy_penalty = (
            1.0 if canonical_path(item.get("path"))[1] or item.get("canonical_of") else 0.0
        )
        prior = tier_prior(item, today) + metadata_adjustment(item, today) - copy_penalty
        item["score"] = base + prior / RRF_K
        item["fusion_debug"] = {
            "scope_ranks": per_scope,
            "rrf_score": base,
            "tier_prior": tier_prior(item, today),
            "metadata_adjustment": metadata_adjustment(item, today),
            "copy_penalty": copy_penalty,
            "duplicates_collapsed": len(group) - 1,
        }
        # Preserve provenance without replacing canonical text or confidence evidence.
        alternatives = list(item.get("alternate_provenance", []))
        for _, _, row in group:
            for provenance in [row.get("provenance"), *row.get("alternate_provenance", [])]:
                if (
                    provenance
                    and provenance != item.get("provenance")
                    and provenance not in alternatives
                ):
                    alternatives.append(provenance)
        item["alternate_provenance"] = alternatives
        results.append(item)
    results.sort(key=lambda item: (-item["score"], str(item.get("id", ""))))
    for rank, item in enumerate(results, 1):
        item["fusion_debug"]["post_fusion_rank"] = rank
    return results
