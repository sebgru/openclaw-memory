import fcntl
import hashlib
import json
import logging
import os
import posixpath
import secrets
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from .doc_metadata import DEFAULT_FRONTMATTER_DATE_KEYS
from .document_indexer import DocumentIndexer, IndexBusyError
from .embeddings import EmbeddingClient
from .facts import FactError, FactStore, authorized, token_from_env
from .fusion import RRF_K, _identities, fuse_scopes
from .indexer import Indexer
from .maintenance import database_availability, verify_database
from .nimble import rerank as nimble_rerank
from .promotion import PromotionError, candidates, promote
from .qdrant import QdrantStore, point_id
from .sqlite_store import SQLiteStore

logger = logging.getLogger("memory_store")

ROOT = os.getenv("DOCUMENT_ROOT", "./documents")
DIM = int(os.getenv("EMBEDDING_DIMENSIONS", "128"))


def patterns(name, default):
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


INCLUDE_PATTERNS = patterns("INCLUDE_PATTERNS", "**/*.md")
DEFAULT_EXCLUDE_PATTERNS = ",".join(
    (
        "memory/dreaming/**",
        "**/memory/dreaming/**",
        "memory/.dreams/**",
        "**/memory/.dreams/**",
        "digests/**",
        "**/digests/**",
        "handoffs/**",
        "**/handoffs/**",
        "system-health/**",
        "**/system-health/**",
        "system-health*.md",
        "**/system-health*.md",
        "review-candidates/**",
        "**/review-candidates/**",
        "archive/**",
        "**/archive/**",
        "exchange/**",
        "**/exchange/**",
    )
)
EXCLUDE_PATTERNS = patterns("EXCLUDE_PATTERNS", DEFAULT_EXCLUDE_PATTERNS)
ARCHIVE_ROOT = os.getenv("ARCHIVE_ROOT")
DOCUMENTS_ROOT = os.getenv("DOCUMENTS_ROOT")
embedder = EmbeddingClient(os.getenv("EMBEDDING_URL"), os.getenv("EMBEDDING_MODEL", "default"), DIM)
store = SQLiteStore(os.getenv("SQLITE_PATH", "memory.db"), DIM)
vector_store = (
    QdrantStore(os.getenv("QDRANT_URL"), os.getenv("QDRANT_COLLECTION", "memory"), DIM)
    if os.getenv("QDRANT_URL")
    else None
)
archive_store = (
    SQLiteStore(os.getenv("ARCHIVE_SQLITE_PATH", "archive.db"), DIM) if ARCHIVE_ROOT else None
)
archive_vector_store = (
    QdrantStore(
        os.getenv("QDRANT_URL"),
        os.getenv("ARCHIVE_QDRANT_COLLECTION", "memory-archive"),
        DIM,
    )
    if ARCHIVE_ROOT and os.getenv("QDRANT_URL")
    else None
)
documents_store = (
    SQLiteStore(
        os.getenv("DOCUMENTS_SQLITE_PATH", "documents.db"),
        DIM,
        enable_fuzzy=False,
        doc_metadata=True,
    )
    if DOCUMENTS_ROOT
    else None
)
documents_vector_store = (
    QdrantStore(
        os.getenv("QDRANT_URL"),
        os.getenv("DOCUMENTS_QDRANT_COLLECTION", "documents"),
        DIM,
    )
    if DOCUMENTS_ROOT and os.getenv("QDRANT_URL")
    else None
)
# Mail is never searched by ordinary all-scope calls. The private-chat adapter must
# explicitly select it and supply a distinct capability token after checking owner identity.
MAIL_PATH = os.getenv("MAIL_SQLITE_PATH")
MAIL_TOKEN = os.getenv("MAIL_ACCESS_TOKEN", "")
if MAIL_PATH and not os.path.isabs(MAIL_PATH):
    raise ValueError("mail database path must be absolute")
if MAIL_PATH and os.path.realpath(MAIL_PATH) in {
    os.path.realpath(os.getenv(name, default))
    for name, default in (
        ("SQLITE_PATH", "memory.db"),
        ("ARCHIVE_SQLITE_PATH", "archive.db"),
        ("DOCUMENTS_SQLITE_PATH", "documents.db"),
    )
}:
    raise ValueError("mail database must be separate from other scopes")
mail_store = SQLiteStore(MAIL_PATH, DIM) if MAIL_PATH and MAIL_TOKEN else None
MAIL_COLLECTION = os.getenv("MAIL_QDRANT_COLLECTION", "mail-private")
if mail_store and (
    not (MAIL_COLLECTION == "mail-private" or MAIL_COLLECTION.startswith("mail-private-"))
    or MAIL_COLLECTION
    in {
        os.getenv("QDRANT_COLLECTION", "memory"),
        os.getenv("ARCHIVE_QDRANT_COLLECTION", "memory-archive"),
        os.getenv("DOCUMENTS_QDRANT_COLLECTION", "documents"),
    }
):
    raise ValueError("mail collection must use a separate mail-private namespace")
mail_vector_store = (
    QdrantStore(os.getenv("QDRANT_URL"), MAIL_COLLECTION, DIM)
    if mail_store and os.getenv("QDRANT_URL")
    else None
)

DOCUMENTS_INCLUDE_PATTERNS = patterns("DOCUMENTS_INCLUDE_PATTERNS", "**/*.md")
DOCUMENTS_EXCLUDE_PATTERNS = patterns("DOCUMENTS_EXCLUDE_PATTERNS", "")
# Default OFF: prepend "title | date | category" to document text before embedding/FTS.
# Only affects files indexed afterwards; see README "Document metadata" for the re-index.
DOCUMENTS_CONTEXT_HEADER = os.getenv("DOCUMENTS_CONTEXT_HEADER", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DOCUMENTS_DATE_FRONTMATTER_KEYS = (
    tuple(
        k.strip().casefold()
        for k in os.getenv("DOCUMENTS_DATE_FRONTMATTER_KEYS", "").split(",")
        if k.strip()
    )
    or DEFAULT_FRONTMATTER_DATE_KEYS
)

FACTS_ENABLED = os.getenv("MEMORY_FACTS_ENABLED", "false").lower() in {"1", "true", "yes"}
FACTS_TOKEN = token_from_env(os.environ) if FACTS_ENABLED else ""
FACTS_PATH = os.getenv("MEMORY_FACTS_SQLITE_PATH", "/data/facts.db")
if FACTS_ENABLED and not os.path.isabs(FACTS_PATH):
    raise ValueError("fact database path must be absolute")
if FACTS_ENABLED and os.path.realpath(FACTS_PATH) in {
    os.path.realpath(os.getenv(name, default))
    for name, default in (
        ("SQLITE_PATH", "memory.db"),
        ("ARCHIVE_SQLITE_PATH", "archive.db"),
        ("DOCUMENTS_SQLITE_PATH", "documents.db"),
    )
}:
    raise ValueError("fact database must be separate from index databases")
fact_store = FactStore(
    FACTS_PATH,
    enabled=FACTS_ENABLED,
    recency_days=int(os.getenv("MEMORY_FACTS_RECENCY_DAYS", "60")) if FACTS_ENABLED else 60,
    timezone=os.getenv("MEMORY_FACTS_TIMEZONE", "UTC") if FACTS_ENABLED else "UTC",
)
if FACTS_ENABLED and not FACTS_TOKEN:
    logger.warning("facts enabled without a token; all fact endpoints fail closed")


_DEFAULT_VECTOR = object()


def env_float(name, default):
    return float(os.getenv(name, str(default)))


def env_int(name, default):
    return int(os.getenv(name, str(default)))


PROMPT_MIN_RELEVANCE = env_float("PROMPT_MIN_RELEVANCE", 0.2)
EXACT_MATCH_BOOST = env_float("EXACT_MATCH_BOOST", 1.0)
TERM_COVERAGE_WEIGHT = env_float("TERM_COVERAGE_WEIGHT", 0.3)
MIN_TERM_LENGTH = 3
# Full coverage of every query term (>= 2 distinctive terms) is treated as a
# strong lexical signal. It must outweigh a semantic-only hit, whose legacy
# score starts at raw cosine similarity (~0.5-0.7), or lexical matches for
# multi-identifier queries are crowded out of the ranking entirely.
FULL_COVERAGE_BOOST = env_float("FULL_COVERAGE_BOOST", 1.0)
# A result set with no lexical evidence and no semantic similarity at or above
# this floor is reported as low confidence (probably nothing relevant exists).
# Typo-tolerant candidates contribute a bounded rank score (<= one lexical rank).
FUZZY_RANK_WEIGHT = min(1.0, env_float("FUZZY_RANK_WEIGHT", 0.5))
FUZZY_MAX_CANDIDATES = env_int("FUZZY_MAX_CANDIDATES", 20)
LOW_CONFIDENCE_SEMANTIC_FLOOR = env_float("LOW_CONFIDENCE_SEMANTIC_FLOOR", 0.6)
PROMPT_SOURCE_QUOTAS = {
    "memory": env_int("PROMPT_MEMORY_QUOTA", 3),
    "artifact": env_int("PROMPT_ARTIFACT_QUOTA", 1),
    "session": env_int("PROMPT_SESSION_QUOTA", 2),
    "archive": env_int("PROMPT_ARCHIVE_QUOTA", 1),
    "document": env_int("PROMPT_DOCUMENT_QUOTA", 2),
}


# Per-thread flags retain hybrid_search's public call/return shape while unified
# search selects rank-only scoring and reports lexical-only fallback.
_search_state = threading.local()


def _current_mail_vectors(selected_store, hits):
    """Reject removed/replaced chunks, even while remote deletion is failing.

    Qdrant payloads are not authoritative. Require the current SQLite chunk's
    point identity and payload, and return SQLite content rather than remote text.
    """
    current = []
    with selected_store._read_connection() as db:
        for hit in hits:
            rows = db.execute(
                "SELECT c.* FROM chunks c JOIN files f ON f.path=c.path WHERE c.path=?",
                (hit.get("path"),),
            )
            for row in rows:
                if hit.get("id") == point_id(row["id"]) and all(
                    hit.get(key) == row[column]
                    for key, column in (("heading", "heading"), ("text", "body"), ("line", "line"))
                ):
                    current.append(
                        {
                            **hit,
                            "path": row["path"],
                            "heading": row["heading"],
                            "text": row["body"],
                            "line": row["line"],
                        }
                    )
                    break
    return current


def hybrid_search(query, limit, selected_store=None, selected_vector_store=_DEFAULT_VECTOR):
    _search_state.lexical_fallback = False
    rank_fusion = getattr(_search_state, "rank_fusion", False)
    selected_store = selected_store or store
    selected_vector_store = (
        vector_store if selected_vector_store is _DEFAULT_VECTOR else selected_vector_store
    )
    lexical = selected_store.search(query, limit * 3)
    fuzzy = (
        selected_store.search_fuzzy(query, min(limit * 3, FUZZY_MAX_CANDIDATES))
        if hasattr(selected_store, "search_fuzzy")
        else []
    )
    if hasattr(selected_store, "search_exact"):
        exact_candidates = selected_store.search_exact(query, min(limit * 2, 100))
        by_id = {row["id"]: row for row in lexical}
        for row in exact_candidates:
            by_id.setdefault(row["id"], row)
        lexical = list(by_id.values())
    try:
        semantic = (
            selected_vector_store.search(embedder.embed(query), limit * 3)
            if selected_vector_store
            else []
        )
    except Exception as exc:
        logger.warning("semantic search unavailable, falling back to FTS (%s)", type(exc).__name__)
        _search_state.lexical_fallback = True
        semantic = []
    if selected_store is mail_store:
        semantic = _current_mail_vectors(selected_store, semantic)
    merged: dict = {}
    max_lexical_evidence = max((max(0.0, row.get("score", 0.0)) for row in lexical), default=0)
    # SQLite rows carry raw chunk IDs while Qdrant hits carry UUID5 point IDs;
    # key lexical rows by their point ID so both backends fuse on one identity.
    for rank, row in enumerate(lexical):
        score = 1 / (RRF_K + rank + 1)
        item = merged.setdefault(
            point_id(row["id"]),
            {
                **row,
                "score": 0,
                "lexical_score": 0,
                "semantic_score": 0,
                "lexical_evidence": 0,
                "semantic_similarity": 0,
            },
        )
        item["score"] += score
        if rank_fusion:
            item.setdefault("engine_ranks", {})["lexical"] = rank + 1
        item["lexical_score"] += score
        item["lexical_evidence"] = max(item["lexical_evidence"], row.get("score", 0))
    # Typo-tolerant candidates are kept distinct from true lexical evidence:
    # they can improve recall/ranking but must not silently turn a weak match
    # into a confident one.
    for rank, row in enumerate(fuzzy):
        # Rank-based, same scale as the lexical/semantic contributions, so a
        # typo candidate can never outweigh an exact or lexical match.
        fuzzy_rank_score = FUZZY_RANK_WEIGHT / (RRF_K + rank + 1)
        item = merged.setdefault(
            point_id(row["id"]),
            {
                **row,
                "score": 0,
                "lexical_score": 0,
                "semantic_score": 0,
                "lexical_evidence": 0,
                "semantic_similarity": 0,
                "fuzzy_match": True,
                "fuzzy_score": fuzzy_rank_score,
            },
        )
        if not item.get("fuzzy_match"):
            item["fuzzy_match"] = True
            item["fuzzy_score"] = fuzzy_rank_score
        item["fuzzy_terms"] = sorted(
            set(item.get("fuzzy_terms", [])) | set(row.get("fuzzy_terms", []))
        )
        if rank_fusion:
            item.setdefault("engine_ranks", {})["fuzzy"] = rank + 1
        item["score"] += fuzzy_rank_score
    for rank, row in enumerate(semantic):
        score = 1 / (RRF_K + rank + 1)
        item = merged.setdefault(
            row["id"],
            {
                **row,
                # Preserve the established ranking contract: a semantic-only
                # result starts with Qdrant's raw similarity and then receives
                # its reciprocal-rank contribution. New relevance fields below
                # make thresholding independent from this legacy score.
                "score": 0 if rank_fusion else row.get("score", 0),
                "lexical_score": 0,
                "semantic_score": 0,
                "lexical_evidence": 0,
                "semantic_similarity": 0,
            },
        )
        item["score"] += score
        if rank_fusion:
            item.setdefault("engine_ranks", {})["vector"] = rank + 1
        item["semantic_score"] += score
        item["semantic_similarity"] = max(item["semantic_similarity"], row.get("score", 0))
    for item in merged.values():
        lexical_relevance = (
            max(0.0, item["lexical_evidence"]) / max_lexical_evidence
            if max_lexical_evidence
            else 0.0
        )
        semantic_relevance = max(0.0, min(1.0, item["semantic_similarity"]))
        item["relevance_score"] = max(lexical_relevance, semantic_relevance)
        bonus, exact = exact_match_bonus(query, item, item.get("fuzzy_terms", []))
        item["exact_match"] = exact
        item["score"] += bonus / (RRF_K + 1) if rank_fusion else bonus
        if rank_fusion:
            item["legacy_candidate_score"] = (
                item["lexical_score"]
                + item["semantic_score"]
                + item.get("fuzzy_score", 0)
                + bonus
                + (
                    item["semantic_similarity"]
                    if not item["lexical_score"] and not item.get("fuzzy_match")
                    else 0
                )
            )
    results = [
        annotate_result(item)
        for item in sorted(
            merged.values(),
            key=(lambda x: (-x["score"], str(x.get("path", "")), str(x["id"])))
            if rank_fusion
            else (lambda x: (-x["score"],)),
        )[:limit]
    ]
    # Stores without document metadata (main/archive, test doubles) return results unchanged.
    if hasattr(selected_store, "attach_metadata"):
        results = selected_store.attach_metadata(results)
    return results


def indexer(root, selected_store, selected_vector_store, includes=("**/*.md",), excludes=()):
    return Indexer(
        root,
        selected_store,
        vector_store=selected_vector_store,
        embed=embedder.embed,
        include_patterns=includes,
        exclude_patterns=excludes,
    )


def vector_status(selected_store, selected_vector_store, bounded=False):
    if not selected_vector_store:
        return None
    if bounded and hasattr(selected_vector_store, "bounded_diagnostics"):
        return selected_vector_store.bounded_diagnostics(selected_store)
    return selected_vector_store.diagnostics(selected_store)


def annotate_result(result):
    """Attach a source label without reading the artifact itself."""
    path = str(result.get("path", ""))
    source = "artifact" if path == "outputs/INDEX.md" or path.startswith("outputs/") else "memory"
    return {**result, "source": source}


def _relaxed_text(value):
    """Normalize punctuation/spacing for identifying verbatim query matches."""
    return " ".join(
        "".join(char if char.isalnum() else " " for char in str(value or "").casefold()).split()
    )


def exact_match_bonus(query, item, fuzzy_terms=()):
    """Promote exact entity/model matches above semantically similar results."""
    needle = _relaxed_text(query)
    if not needle:
        return 0.0, False
    haystack = _relaxed_text(f"{item.get('heading', '')} {item.get('text', '')}")
    path_text = _relaxed_text(item.get("path", ""))
    if len(needle) >= MIN_TERM_LENGTH and (needle in haystack or needle in path_text):
        return EXACT_MATCH_BOOST, True
    terms = [term for term in needle.split() if len(term) >= MIN_TERM_LENGTH]
    if not terms:
        return 0.0, False
    fuzzy_terms = set(fuzzy_terms)
    matched = sum(
        1 for term in terms if term in haystack or term in path_text or term in fuzzy_terms
    )
    if matched == len(terms) and len(terms) >= 2:
        return FULL_COVERAGE_BOOST, False
    return TERM_COVERAGE_WEIGHT * matched / len(terms), False


def is_low_confidence(results):
    """True when results carry no lexical or exact evidence and weak semantic similarity.

    Results lacking the relevance fields (unknown) are never reported as low
    confidence. An empty result list is not "low confidence"; it is just empty.
    """
    if not results or any("semantic_similarity" not in item for item in results):
        return False
    for item in results:
        if item.get("lexical_evidence", 0) > 0 or item.get("exact_match"):
            return False
        if item["semantic_similarity"] >= LOW_CONFIDENCE_SEMANTIC_FLOOR:
            return False
    return True


def unified_source(result, archive=False, document=False):
    """Return the stable corpus label for a unified-search result."""
    if archive:
        return "archive"
    if document:
        return "document"
    path = str(result.get("path", ""))
    if path.startswith(("sessions/", "session/")) or "/sessions/" in path:
        return "session"
    if path == "outputs/INDEX.md" or path.startswith("outputs/"):
        return "artifact"
    return "memory"


def unified_result_id(result, source):
    """Build an ID independent of SQLite or Qdrant's backend-specific IDs."""
    normalized_text = " ".join(str(result.get("text", "")).casefold().split())
    if source in {"session", "archive"}:
        # Imported session corpora may contain the same transcript under several
        # provenance paths. Content identity prevents duplicate copies from
        # dominating recall while provenance remains attached below.
        identity = "\0".join(("conversation", normalized_text))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()
    identity = "\0".join(
        (
            source,
            str(result.get("path", "")),
            str(result.get("line", "")),
            str(result.get("heading", "")),
            str(result.get("text", "")),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def result_provenance(result, source):
    return {
        "source": source,
        "path": str(result.get("path", "")),
        "heading": result.get("heading"),
        "line": result.get("line"),
    }


def apply_search_profile(results, limit, profile):
    if profile == "tool":
        return results[:limit]
    selected = []
    counts = {source: 0 for source in PROMPT_SOURCE_QUOTAS}
    for item in results:
        source = item["source"]
        if item.get("relevance_score", 0) < PROMPT_MIN_RELEVANCE:
            continue
        quota = PROMPT_SOURCE_QUOTAS.get(source, 0)
        if counts.get(source, 0) >= quota:
            continue
        counts[source] = counts.get(source, 0) + 1
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


COVERAGE_SOURCES = ("main", "archive", "documents")


def unified_search(query, limit, scope="all", profile="tool"):
    """Search the configured indexes; return ``(results, warnings)``."""
    results, warnings, *_ = unified_search_with_coverage(query, limit, scope, profile)
    return results, warnings


def unified_search_with_coverage(query, limit, scope="all", profile="tool", *, debug=False):
    """Search the configured indexes and report what was actually searched.

    Returns ``(results, warnings, coverage, degraded)``. ``degraded`` maps a searched
    source to ``lexical_fallback`` when semantic search failed and only FTS ran.
    ``coverage`` maps each of
    main/archive/documents to ``searched`` (backend ran without error),
    ``unavailable`` (requested by scope but not configured or failed) or
    ``not_searched`` (excluded by scope).
    """
    if scope == "mail":
        if profile not in {"tool", "prompt"}:
            raise ValueError("profile must be one of tool, prompt")
        if mail_store is None:
            raise LookupError("mail is not configured")
        coverage = dict.fromkeys((*COVERAGE_SOURCES, "mail"), "not_searched")
        mode = os.getenv("UNIFIED_FUSION_MODE", "legacy").strip().lower()
        if mode not in {"legacy", "rrf"}:
            raise ValueError("UNIFIED_FUSION_MODE must be legacy or rrf")
        previous_mode = getattr(_search_state, "rank_fusion", False)
        try:
            _search_state.rank_fusion = mode == "rrf"
            try:
                rows = hybrid_search(query, limit * 3, mail_store, mail_vector_store)
            finally:
                _search_state.rank_fusion = previous_mode
        except Exception:
            coverage["mail"] = "unavailable"
            return [], ["mail: search backend unavailable"], coverage, {}
        coverage["mail"] = "searched"
        results = [
            {
                **row,
                "source": "mail",
                "id": unified_result_id(row, "mail"),
                "provenance": result_provenance(row, "mail"),
            }
            for row in rows
        ]
        # One result per message, not three chunks from one long mail.
        unique: dict[str, dict] = {}
        for row in results:
            unique.setdefault(row["path"], row)
        results = list(unique.values())[:limit]
        with mail_store._read_connection() as db:
            metadata_exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mail_metadata'"
            ).fetchone()
            if metadata_exists:
                for row in results:
                    meta = db.execute(
                        "SELECT subject, sender, date FROM mail_metadata WHERE path=?",
                        (row["path"],),
                    ).fetchone()
                    if meta:
                        row["mail"] = dict(meta)
        fallback = mail_vector_store is None or getattr(_search_state, "lexical_fallback", False)
        warnings = (
            ["mail: semantic search unavailable, lexical (FTS) fallback used"] if fallback else []
        )
        return results, warnings, coverage, {"mail": "lexical_fallback"} if fallback else {}
    if scope not in {"all", "main", "archive", "documents"}:
        raise ValueError("scope must be one of all, main, archive, documents, mail")
    if profile not in {"tool", "prompt"}:
        raise ValueError("profile must be one of tool, prompt")
    if scope == "archive" and not archive_store:
        raise LookupError("archive is not configured")
    if scope == "documents" and not documents_store:
        raise LookupError("documents are not configured")

    mode = os.getenv("UNIFIED_FUSION_MODE", "legacy").strip().lower()
    if mode not in {"legacy", "rrf"}:
        raise ValueError("UNIFIED_FUSION_MODE must be legacy or rrf")
    lanes = []
    warnings = []
    merged: dict[str, dict] = {}
    backends = []
    coverage = dict.fromkeys(COVERAGE_SOURCES, "not_searched")
    degraded: dict[str, str] = {}
    if scope in {"all", "main"}:
        backends.append(("main", store, vector_store, False, False))
    if scope in {"all", "archive"}:
        if archive_store:
            backends.append(("archive", archive_store, archive_vector_store, True, False))
        else:
            warnings.append("archive: not configured")
            coverage["archive"] = "unavailable"

    if scope in {"all", "documents"}:
        if documents_store:
            backends.append(("documents", documents_store, documents_vector_store, False, True))
        else:
            warnings.append("documents: not configured")
            coverage["documents"] = "unavailable"

    for name, selected_store, selected_vector_store, is_archive, is_document in backends:
        _search_state.lexical_fallback = False
        try:
            previous_mode = getattr(_search_state, "rank_fusion", False)
            _search_state.rank_fusion = mode == "rrf"
            try:
                rows = hybrid_search(query, limit * 3, selected_store, selected_vector_store)
            finally:
                _search_state.rank_fusion = previous_mode
        except Exception as exc:
            logger.warning("unified %s search unavailable (%s)", name, type(exc).__name__)
            warnings.append(f"{name}: search backend unavailable")
            coverage[name] = "unavailable"
            continue
        coverage[name] = "searched"
        if getattr(_search_state, "lexical_fallback", False):
            degraded[name] = "lexical_fallback"
            # Searched successfully, but without semantic matching; say so.
            warnings.append(f"{name}: semantic search unavailable, lexical (FTS) fallback used")
        lane = []
        for row in rows:
            source = unified_source(row, archive=is_archive, document=is_document)
            result_id = unified_result_id(row, source)
            if mode == "rrf":
                lane.append(
                    {
                        **row,
                        "id": result_id,
                        "source": source,
                        "provenance": result_provenance(row, source),
                        "alternate_provenance": [],
                    }
                )
                continue
            item = merged.setdefault(
                result_id,
                {
                    **row,
                    "id": result_id,
                    "source": source,
                    "score": 0,
                    "lexical_score": 0,
                    "semantic_score": 0,
                    "lexical_evidence": 0,
                    "semantic_similarity": 0,
                    "relevance_score": 0,
                    "provenance": result_provenance(row, source),
                    "alternate_provenance": [],
                },
            )
            provenance = result_provenance(row, source)
            same_provenance = provenance == item["provenance"]
            if not same_provenance and provenance not in item["alternate_provenance"]:
                item["alternate_provenance"].append(provenance)
            combine = (lambda old, new: old + new) if same_provenance else max
            for field in ("score", "lexical_score", "semantic_score"):
                item[field] = combine(item[field], row.get(field, 0))
            for field in ("lexical_evidence", "semantic_similarity", "relevance_score"):
                item[field] = max(item[field], row.get(field, 0))

        if mode == "rrf":
            lanes.append((name, lane))

    if mode == "rrf":
        # Nimble supplies an optional within-scope rank, never a cross-scope score.
        # One bounded call retains its request deadline/candidate cap.
        # Interleave scope ranks so the bounded prefix is not main-only.
        # Dedup before Nimble so copies do not consume its bounded prefix.
        identity_cache = {id(row): _identities(row) for _, lane in lanes for row in lane}
        legacy_ranks: dict[tuple, int] = {}
        if debug:
            legacy_candidates = sorted(
                [row for _, lane in lanes for row in lane],
                key=lambda row: -row.get("legacy_candidate_score", row["score"]),
            )
            for rank, row in enumerate(legacy_candidates, 1):
                for key in identity_cache[id(row)]:
                    legacy_ranks.setdefault(key, rank)
        group_members: dict[int, list[dict]] = {}
        canonical = fuse_scopes(lanes, identity_cache=identity_cache, group_members=group_members)
        # Every original member (including transitive identity bridges) maps to
        # its group explicitly; canonical keys alone cannot identify all members.
        owner = {
            id(member): id(group) for group in canonical for member in group_members[id(group)]
        }
        canonical_lanes = []
        for name, _ in lanes:
            lane = [row for row in canonical if name in row["fusion_debug"]["scope_ranks"]]
            lane.sort(key=lambda row: row["fusion_debug"]["scope_ranks"][name])
            canonical_lanes.append((name, lane))
        ranked = [(rank, row) for _, lane in canonical_lanes for rank, row in enumerate(lane)]
        candidates = []
        seen = set()
        for _, row in sorted(ranked, key=lambda pair: pair[0]):
            if id(row) not in seen:
                candidates.append(row)
                seen.add(id(row))
        reranked = nimble_rerank(query, candidates)
        if [id(row) for row in reranked] == [id(row) for row in candidates]:
            # Disabled, fallback, and no-op reranking must preserve true ranks.
            # Reuse the first fusion rather than repeating identity/group work.
            results = canonical
        else:
            order = {id(row): rank for rank, row in enumerate(reranked)}
            moved = {id(row) for rank, row in enumerate(candidates) if order[id(row)] != rank}
            reordered = []
            for name, lane in lanes:
                # Reorder only moved groups represented in THIS lane, within
                # their existing slots. Unaffected members retain their original
                # relative order; a different lane's rank is never borrowed.
                slots = [i for i, row in enumerate(lane) if owner[id(row)] in moved]
                changed = sorted((lane[i] for i in slots), key=lambda row: order[owner[id(row)]])
                updated = list(lane)
                for i, row in zip(slots, changed, strict=True):
                    updated[i] = row
                reordered.append((name, updated))
            results = fuse_scopes(reordered, identity_cache=identity_cache)
        if debug:
            for item in results:
                item["fusion_debug"]["legacy_candidate_rank"] = min(
                    (legacy_ranks[key] for key in _identities(item) if key in legacy_ranks),
                    default=None,
                )
    else:
        results = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
        results = nimble_rerank(query, results)
    # Legacy/main evaluates only delivered rows; RRF keeps query-level evidence.
    _search_state.low_confidence = is_low_confidence(results) if mode == "rrf" else None
    selected = apply_search_profile(results, limit, profile)
    for rank, item in enumerate(selected, 1):
        if debug and "fusion_debug" in item:
            item["fusion_debug"]["post_profile_rank"] = rank
        item.pop("legacy_candidate_score", None)
        item.pop("engine_ranks", None)
        if not debug:
            item.pop("fusion_debug", None)
    return selected, warnings, coverage, degraded


def private_fact_log_path(target):
    """Classify raw and decoded paths before normalization can remove private prefixes."""
    try:
        parsed = urlparse(target)
        raw_path = target.split("?", 1)[0]
        decoded_path = unquote(parsed.path)
        path = "/" + posixpath.normpath(decoded_path).lstrip("/")
        paths = (raw_path, unquote(raw_path), parsed.path, decoded_path, path)
        keys = parse_qs(parsed.query, keep_blank_values=True)
    except ValueError:
        # Malformed targets are not safe to echo into access logs either.
        return "/"
    if any(candidate == "/facts" or candidate.startswith("/facts/") for candidate in paths) or any(
        key.casefold().startswith("fact_") for key in keys
    ):
        known_paths = {
            "/facts/drafts",
            "/facts/confirm",
            "/facts/end",
            "/facts/retract",
            "/facts/current",
            "/unified/search",
        }
        return path if path in known_paths else "/facts/"
    return None


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def facts_authorized(self):
        if not FACTS_TOKEN:
            self.send_json(503, {"error": "fact authentication unavailable"})
            return False
        if not authorized(self.headers.get("Authorization"), FACTS_TOKEN):
            self.send_json(401, {"error": "fact authentication required"})
            return False
        return True

    def current_fact(self, params):
        subject = params.get("subject", [""])[0]
        predicate = params.get("predicate", [""])[0]
        if not subject or not predicate:
            raise FactError("subject and predicate are required")
        return fact_store.read(
            subject,
            predicate,
            params.get("item_key", [None])[0],
            params.get("as_of", [None])[0],
            params.get("include_history", ["false"])[0] == "true",
        )

    def facts_post(self):
        if not self.facts_authorized():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise FactError("invalid fact request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise FactError("invalid fact request")
            if self.path == "/facts/confirm":
                if set(payload) != {"draft_id", "confirmation"}:
                    raise FactError("draft_id and confirmation are required")
                if not isinstance(payload["draft_id"], str):
                    raise FactError("invalid draft id")
                result = fact_store.confirm(payload["draft_id"], payload["confirmation"])
            else:
                expected = {"/facts/end": "end", "/facts/retract": "retract"}.get(self.path)
                if expected and payload.get("action") != expected:
                    raise FactError("action does not match endpoint")
                result = fact_store.draft(payload)
            return self.send_json(200, result)
        except (ValueError, TypeError, KeyError, RecursionError):
            return self.send_json(400, {"error": "invalid or expired fact capture"})
        except (OSError, sqlite3.Error):
            return self.send_json(503, {"error": "fact store unavailable"})

    def do_GET(self):
        started = time.monotonic()
        parsed, params = urlparse(self.path), parse_qs(urlparse(self.path).query)
        if FACTS_ENABLED and parsed.path == "/facts/current":
            if not self.facts_authorized():
                return
            try:
                return self.send_json(200, self.current_fact(params))
            except FactError:
                return self.send_json(400, {"error": "invalid fact lookup"})
            except (OSError, sqlite3.Error):
                return self.send_json(
                    503,
                    {
                        "subject": params.get("subject", [""])[0],
                        "predicate": params.get("predicate", [""])[0],
                        "state": "unavailable",
                        "disclosures": ["partial_lane_unavailable"],
                    },
                )
        if parsed.path == "/healthz":
            availability = database_availability(
                store.db.execute("PRAGMA database_list").fetchone()[2]
            )
            return self.send_json(
                200,
                {
                    "status": availability["status"],
                    "backend": "sqlite+qdrant" if vector_store else "sqlite",
                    "database": availability,
                    **store.status(quick=True),
                },
            )
        if parsed.path == "/status":
            integrity = verify_database(store.db.execute("PRAGMA database_list").fetchone()[2])
            status = {
                "status": integrity["status"],
                "backend": "sqlite+qdrant" if vector_store else "sqlite",
                "database": integrity,
                **store.status(),
            }
            if vector_store:
                try:
                    status["vectors"] = vector_status(store, vector_store)
                except Exception as exc:
                    status["vectors"] = {"status": "error", "error": str(exc)}
            return self.send_json(
                200,
                status,
            )
        if parsed.path == "/documents/status":
            if not documents_store:
                return self.send_json(404, {"error": "documents are not configured"})
            try:
                integrity = database_availability(
                    documents_store.path, documents_store.read_timeout
                )
                document_status = DocumentIndexer(
                    DOCUMENTS_ROOT,
                    documents_store,
                    documents_vector_store,
                    embedder.embed,
                    include_patterns=DOCUMENTS_INCLUDE_PATTERNS,
                    exclude_patterns=DOCUMENTS_EXCLUDE_PATTERNS,
                    read_only=True,
                ).status(quick=True)
            except sqlite3.OperationalError as exc:
                logger.warning("%s unavailable: %s", parsed.path, exc)
                return self.send_json(503, {"error": "documents database busy", "detail": str(exc)})
            status = {"status": integrity["status"], "database": integrity, **document_status}
            if documents_vector_store:
                try:
                    status["vectors"] = vector_status(
                        documents_store, documents_vector_store, bounded=True
                    )
                except Exception as exc:
                    status["vectors"] = {"status": "error", "error": str(exc)}
            return self.send_json(200, status)
        if parsed.path == "/documents/parity":
            if not documents_store:
                return self.send_json(404, {"error": "documents are not configured"})
            if not documents_vector_store:
                return self.send_json(404, {"error": "vector store is not configured"})
            try:
                limit = int(params.get("limit", [256])[0])
            except ValueError:
                return self.send_json(400, {"error": "limit must be an integer"})
            if not 1 <= limit <= 1000:
                return self.send_json(400, {"error": "limit must be between 1 and 1000"})
            after = params.get("after", [""])[0]
            if len(after) > 1024 or "\x00" in after:
                return self.send_json(400, {"error": "after is not a valid cursor"})
            try:
                page = documents_vector_store.parity_page(documents_store, after, limit)
            except sqlite3.OperationalError as exc:
                logger.warning("%s unavailable: %s", parsed.path, exc)
                return self.send_json(503, {"error": "documents database busy", "detail": str(exc)})
            except Exception as exc:
                return self.send_json(
                    502, {"error": "vector store unavailable", "detail": str(exc)}
                )
            return self.send_json(200, {"status": "ok", "parity": "page", **page})
        if parsed.path == "/documents/errors":
            if not documents_store:
                return self.send_json(404, {"error": "documents are not configured"})
            try:
                limit = int(params.get("limit", [100])[0])
            except ValueError:
                return self.send_json(400, {"error": "limit must be an integer"})
            if not 1 <= limit <= 1000:
                return self.send_json(400, {"error": "limit must be between 1 and 1000"})
            indexer = DocumentIndexer(
                DOCUMENTS_ROOT,
                documents_store,
                documents_vector_store,
                embedder.embed,
                include_patterns=DOCUMENTS_INCLUDE_PATTERNS,
                exclude_patterns=DOCUMENTS_EXCLUDE_PATTERNS,
                read_only=True,
            )
            try:
                return self.send_json(200, {"errors": indexer.errors(limit=limit)})
            except sqlite3.OperationalError as exc:
                logger.warning("%s unavailable: %s", parsed.path, exc)
                return self.send_json(503, {"error": "documents database busy", "detail": str(exc)})
        if parsed.path == "/promotion/candidates":
            return self.send_json(200, {"candidates": candidates(ROOT)})
        if parsed.path == "/archive/status":
            if not archive_store:
                return self.send_json(404, {"error": "archive is not configured"})
            integrity = database_availability(
                archive_store.database_path, archive_store.read_timeout
            )
            with archive_store._read_connection() as db:
                archive_status = archive_store.status(db, include_fts_diagnostics=False, quick=True)
            status = {
                "status": integrity["status"],
                "database": integrity,
                **archive_status,
            }
            if archive_vector_store:
                try:
                    status["vectors"] = vector_status(
                        archive_store, archive_vector_store, bounded=True
                    )
                except Exception as exc:
                    status["vectors"] = {"status": "error", "error": str(exc)}
            return self.send_json(200, status)
        if parsed.path in ("/search", "/archive/search", "/documents/search"):
            is_archive = parsed.path.startswith("/archive/")
            is_documents = parsed.path.startswith("/documents/")
            if is_archive and not archive_store:
                return self.send_json(404, {"error": "archive is not configured"})
            if is_documents and not documents_store:
                return self.send_json(404, {"error": "documents are not configured"})
            query = params.get("q", [""])[0]
            try:
                limit = int(params.get("limit", [10])[0])
            except ValueError:
                return self.send_json(400, {"error": "limit must be an integer"})
            if not query.strip() or not 1 <= limit <= 100:
                return self.send_json(
                    400, {"error": "q is required and limit must be between 1 and 100"}
                )
            try:
                results = hybrid_search(
                    query,
                    limit,
                    archive_store if is_archive else documents_store if is_documents else store,
                    archive_vector_store
                    if is_archive
                    else documents_vector_store
                    if is_documents
                    else vector_store,
                )
                if is_documents:
                    results = [{**row, "source": "document"} for row in results]
                logger.info(
                    "%s limit=%d results=%d took=%.1fms",
                    parsed.path,
                    limit,
                    len(results),
                    (time.monotonic() - started) * 1000,
                )
                return self.send_json(200, {"results": results})
            except Exception as exc:
                logger.exception("%s failed", parsed.path)
                return self.send_json(
                    503, {"error": "search backend unavailable", "detail": str(exc)}
                )
        if parsed.path == "/unified/search":
            current_state = None
            if FACTS_ENABLED and ("fact_subject" in params or "fact_predicate" in params):
                if not self.facts_authorized():
                    return
                fact_params = {
                    key.removeprefix("fact_"): value
                    for key, value in params.items()
                    if key.startswith("fact_")
                }
                try:
                    current_state = [self.current_fact(fact_params)]
                except FactError:
                    return self.send_json(400, {"error": "invalid fact lookup"})
                except (OSError, sqlite3.Error):
                    current_state = [
                        {
                            "subject": fact_params.get("subject", [""])[0],
                            "predicate": fact_params.get("predicate", [""])[0],
                            "state": "unavailable",
                            "disclosures": ["partial_lane_unavailable"],
                        }
                    ]
            query = params.get("q", [""])[0]
            scope = params.get("scope", ["all"])[0]
            profile = params.get("profile", ["tool"])[0]
            if scope == "mail" and (
                not MAIL_TOKEN
                or not secrets.compare_digest(
                    self.headers.get("X-Mail-Access", "").encode(), MAIL_TOKEN.encode()
                )
            ):
                return self.send_json(403, {"error": "mail access denied"})
            try:
                limit = int(params.get("limit", [10])[0])
            except ValueError:
                return self.send_json(400, {"error": "limit must be an integer"})
            if not query.strip() or not 1 <= limit <= 100:
                return self.send_json(
                    400, {"error": "q is required and limit must be between 1 and 100"}
                )
            if scope not in {"all", "main", "archive", "documents", "mail"}:
                return self.send_json(
                    400, {"error": "scope must be one of all, main, archive, documents, mail"}
                )
            if profile not in {"tool", "prompt"}:
                return self.send_json(400, {"error": "profile must be one of tool, prompt"})
            _search_state.low_confidence = None
            debug = params.get("debug", ["0"])[0] == "1"
            try:
                # Keep the long-standing three-argument call shape for the
                # default profile; callers and test doubles remain compatible.
                results, warnings, coverage, degraded = (
                    unified_search_with_coverage(query, limit, scope)
                    if profile == "tool" and not debug
                    else unified_search_with_coverage(query, limit, scope, profile, debug=debug)
                )
            except ValueError:
                logger.warning("invalid unified search configuration")
                return self.send_json(500, {"error": "invalid unified search configuration"})
            except LookupError as exc:
                return self.send_json(404, {"error": str(exc)})
            logger.info(
                "/unified/search scope=%s profile=%s limit=%d results=%d warnings=%d took=%.1fms",
                scope,
                profile,
                limit,
                len(results),
                len(warnings),
                (time.monotonic() - started) * 1000,
            )
            if not debug:
                results = [
                    {key: value for key, value in item.items() if key != "fusion_debug"}
                    for item in results
                ]
            payload: dict = {"current_state": current_state} if current_state is not None else {}
            payload.update(
                results=results,
                coverage=coverage,
                fusion=os.getenv("UNIFIED_FUSION_MODE", "legacy").strip().lower(),
            )
            if degraded:
                payload["degraded"] = degraded
            if warnings:
                payload["warnings"] = warnings
            confidence = getattr(_search_state, "low_confidence", None)
            if confidence if confidence is not None else is_low_confidence(results):
                payload["low_confidence"] = True
                payload["confidence_note"] = (
                    "no lexical match and weak semantic similarity; "
                    "results are probably not relevant to the query"
                )
            return self.send_json(200, payload)
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if FACTS_ENABLED and self.path in {
            "/facts/drafts",
            "/facts/confirm",
            "/facts/end",
            "/facts/retract",
        }:
            return self.facts_post()
        if self.path in ("/reconcile", "/archive/reconcile", "/documents/reconcile"):
            is_archive = self.path.startswith("/archive/")
            is_documents = self.path.startswith("/documents/")
            selected_store = (
                archive_store if is_archive else documents_store if is_documents else store
            )
            selected_vectors = (
                archive_vector_store
                if is_archive
                else documents_vector_store
                if is_documents
                else vector_store
            )
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if payload.get("confirm") is not True:
                    return self.send_json(400, {"error": "reconciliation requires confirm=true"})
                if not selected_store or not selected_vectors:
                    return self.send_json(404, {"error": "vector backend is not configured"})
                db_path = selected_store.db.execute("PRAGMA database_list").fetchone()[2]
                with open(db_path + ".vector-reconcile.lock", "w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    limit = int(payload.get("limit", 1000))
                    if not 1 <= limit <= 10000:
                        return self.send_json(400, {"error": "limit must be between 1 and 10000"})
                    result = selected_vectors.reconcile_missing(
                        selected_store, embedder.embed, limit
                    )
                return self.send_json(200, result)
            except (ValueError, OSError) as exc:
                return self.send_json(400, {"error": str(exc)})
        if self.path == "/promotion/promote":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                result = promote(
                    ROOT, payload["candidate"], payload["destination"], payload["approved_by"]
                )
                return self.send_json(200, result)
            except (KeyError, TypeError, json.JSONDecodeError, PromotionError) as exc:
                return self.send_json(400, {"error": str(exc)})
        if self.path == "/documents/index":
            if not documents_store:
                return self.send_json(404, {"error": "documents are not configured"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                max_files = int(payload.get("max_files", 250))
                reindex = payload.get("reindex", False)
                if not isinstance(reindex, bool):
                    raise ValueError("reindex must be a boolean")
                document_indexer = DocumentIndexer(
                    DOCUMENTS_ROOT,
                    documents_store,
                    documents_vector_store,
                    embedder.embed,
                    include_patterns=DOCUMENTS_INCLUDE_PATTERNS,
                    exclude_patterns=DOCUMENTS_EXCLUDE_PATTERNS,
                    context_header=DOCUMENTS_CONTEXT_HEADER,
                    frontmatter_date_keys=DOCUMENTS_DATE_FRONTMATTER_KEYS,
                )
                # Keep the long-standing call shape unless a re-index is requested.
                stats = (
                    document_indexer.scan(max_files=max_files, reindex=True)
                    if reindex
                    else document_indexer.scan(max_files=max_files)
                )
                logger.info("%s %s", self.path, stats.as_dict())
                return self.send_json(200, stats.as_dict())
            except IndexBusyError as exc:
                return self.send_json(409, {"error": str(exc)})
            except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as exc:
                logger.warning("%s failed: %s", self.path, exc)
                return self.send_json(400, {"error": str(exc)})
        if self.path not in ("/index", "/archive/index"):
            return self.send_json(404, {"error": "not found"})
        is_archive = self.path == "/archive/index"
        if is_archive and not archive_store:
            return self.send_json(404, {"error": "archive is not configured"})
        try:
            stats = indexer(
                ARCHIVE_ROOT if is_archive else ROOT,
                archive_store if is_archive else store,
                archive_vector_store if is_archive else vector_store,
                ("**/*.md",) if is_archive else INCLUDE_PATTERNS,
                () if is_archive else EXCLUDE_PATTERNS,
            ).scan()
            logger.info("%s %s", self.path, stats.as_dict())
            return self.send_json(200, stats.as_dict())
        except (OSError, UnicodeError, ValueError) as exc:
            logger.warning("%s failed: %s", self.path, exc)
            return self.send_json(400, {"error": str(exc)})

    def log_request(self, code="-", size="-"):
        # BaseHTTPRequestHandler otherwise logs the complete request/query.
        # Malformed request lines can reach this before path/command are set.
        if not hasattr(self, "path"):
            logger.info("request rejected before route parsing")
            return
        # BaseHTTPRequestHandler may rewrite self.path; classify the original
        # target so private fact lookups always collapse to their route bucket.
        request_parts = getattr(self, "requestline", "").split()
        target = request_parts[1] if len(request_parts) > 1 else self.path
        fact_path = private_fact_log_path(target)
        if fact_path is not None:
            logger.info("private fact request path=%s", fact_path)
            return
        path = getattr(self, "path", "").split("?", 1)[0]
        logger.info("http path=%s status=%s size=%s", path, code, size)

    def log_error(self, format, *args):
        # Parser errors interpolate the raw request line, including query secrets.
        logger.warning("http request rejected")

    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "starting server on port %s "
        "(document_root=%s, qdrant=%s, archive_root=%s, documents_root=%s)",
        os.getenv("PORT", "8080"),
        ROOT,
        bool(vector_store),
        ARCHIVE_ROOT or "disabled",
        DOCUMENTS_ROOT or "disabled",
    )
    ThreadingHTTPServer(("0.0.0.0", int(os.getenv("PORT", "8080"))), Handler).serve_forever()
