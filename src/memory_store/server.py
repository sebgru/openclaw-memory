import fcntl
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .document_indexer import DocumentIndexer, IndexBusyError
from .embeddings import EmbeddingClient
from .indexer import Indexer
from .maintenance import database_availability, verify_database
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
    SQLiteStore(os.getenv("DOCUMENTS_SQLITE_PATH", "documents.db"), DIM) if DOCUMENTS_ROOT else None
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
DOCUMENTS_INCLUDE_PATTERNS = patterns("DOCUMENTS_INCLUDE_PATTERNS", "**/*.md")
DOCUMENTS_EXCLUDE_PATTERNS = patterns("DOCUMENTS_EXCLUDE_PATTERNS", "")


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
LOW_CONFIDENCE_SEMANTIC_FLOOR = env_float("LOW_CONFIDENCE_SEMANTIC_FLOOR", 0.6)
PROMPT_SOURCE_QUOTAS = {
    "memory": env_int("PROMPT_MEMORY_QUOTA", 3),
    "artifact": env_int("PROMPT_ARTIFACT_QUOTA", 1),
    "session": env_int("PROMPT_SESSION_QUOTA", 2),
    "archive": env_int("PROMPT_ARCHIVE_QUOTA", 1),
    "document": env_int("PROMPT_DOCUMENT_QUOTA", 2),
}


# Per-thread flag so unified search can report a lexical-only fallback without
# changing hybrid_search's return shape.
_search_state = threading.local()


def hybrid_search(query, limit, selected_store=None, selected_vector_store=_DEFAULT_VECTOR):
    _search_state.lexical_fallback = False
    selected_store = selected_store or store
    selected_vector_store = (
        vector_store if selected_vector_store is _DEFAULT_VECTOR else selected_vector_store
    )
    lexical = selected_store.search(query, limit * 3)
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
        logger.warning("semantic search unavailable, falling back to FTS: %s", exc)
        _search_state.lexical_fallback = True
        semantic = []
    merged: dict = {}
    max_lexical_evidence = max((max(0.0, row.get("score", 0.0)) for row in lexical), default=0)
    # SQLite rows carry raw chunk IDs while Qdrant hits carry UUID5 point IDs;
    # key lexical rows by their point ID so both backends fuse on one identity.
    for rank, row in enumerate(lexical):
        score = 1 / (60 + rank + 1)
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
        item["lexical_score"] += score
        item["lexical_evidence"] = max(item["lexical_evidence"], row.get("score", 0))
    for rank, row in enumerate(semantic):
        score = 1 / (60 + rank + 1)
        item = merged.setdefault(
            row["id"],
            {
                **row,
                # Preserve the established ranking contract: a semantic-only
                # result starts with Qdrant's raw similarity and then receives
                # its reciprocal-rank contribution. New relevance fields below
                # make thresholding independent from this legacy score.
                "score": row.get("score", 0),
                "lexical_score": 0,
                "semantic_score": 0,
                "lexical_evidence": 0,
                "semantic_similarity": 0,
            },
        )
        item["score"] += score
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
        bonus, exact = exact_match_bonus(query, item)
        item["exact_match"] = exact
        item["score"] += bonus
    return [
        annotate_result(item)
        for item in sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:limit]
    ]


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


def exact_match_bonus(query, item):
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
    matched = sum(1 for term in terms if term in haystack or term in path_text)
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


def unified_search_with_coverage(query, limit, scope="all", profile="tool"):
    """Search the configured indexes and report what was actually searched.

    Returns ``(results, warnings, coverage, degraded)``. ``degraded`` maps a searched
    source to ``lexical_fallback`` when semantic search failed and only FTS ran.
    ``coverage`` maps each of
    main/archive/documents to ``searched`` (backend ran without error),
    ``unavailable`` (requested by scope but not configured or failed) or
    ``not_searched`` (excluded by scope).
    """
    if scope not in {"all", "main", "archive", "documents"}:
        raise ValueError("scope must be one of all, main, archive, documents")
    if profile not in {"tool", "prompt"}:
        raise ValueError("profile must be one of tool, prompt")
    if scope == "archive" and not archive_store:
        raise LookupError("archive is not configured")
    if scope == "documents" and not documents_store:
        raise LookupError("documents are not configured")

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
            rows = hybrid_search(query, limit * 3, selected_store, selected_vector_store)
        except Exception as exc:
            logger.warning("unified %s search unavailable: %s", name, exc)
            warnings.append(f"{name}: search backend unavailable")
            coverage[name] = "unavailable"
            continue
        coverage[name] = "searched"
        if getattr(_search_state, "lexical_fallback", False):
            degraded[name] = "lexical_fallback"
            # Searched successfully, but without semantic matching; say so.
            warnings.append(f"{name}: semantic search unavailable, lexical (FTS) fallback used")
        for row in rows:
            source = unified_source(row, archive=is_archive, document=is_document)
            result_id = unified_result_id(row, source)
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

    results = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    return apply_search_profile(results, limit, profile), warnings, coverage, degraded


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        started = time.monotonic()
        parsed, params = urlparse(self.path), parse_qs(urlparse(self.path).query)
        if parsed.path in ("/healthz", "/status"):
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
                    "%s q=%r limit=%d results=%d took=%.1fms",
                    parsed.path,
                    query,
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
            query = params.get("q", [""])[0]
            scope = params.get("scope", ["all"])[0]
            profile = params.get("profile", ["tool"])[0]
            try:
                limit = int(params.get("limit", [10])[0])
            except ValueError:
                return self.send_json(400, {"error": "limit must be an integer"})
            if not query.strip() or not 1 <= limit <= 100:
                return self.send_json(
                    400, {"error": "q is required and limit must be between 1 and 100"}
                )
            if scope not in {"all", "main", "archive", "documents"}:
                return self.send_json(
                    400, {"error": "scope must be one of all, main, archive, documents"}
                )
            if profile not in {"tool", "prompt"}:
                return self.send_json(400, {"error": "profile must be one of tool, prompt"})
            try:
                # Keep the long-standing three-argument call shape for the
                # default profile; callers and test doubles remain compatible.
                results, warnings, coverage, degraded = (
                    unified_search_with_coverage(query, limit, scope)
                    if profile == "tool"
                    else unified_search_with_coverage(query, limit, scope, profile)
                )
            except LookupError as exc:
                return self.send_json(404, {"error": str(exc)})
            logger.info(
                "/unified/search q=%r scope=%s profile=%s limit=%d results=%d warnings=%d took=%.1fms",
                query,
                scope,
                profile,
                limit,
                len(results),
                len(warnings),
                (time.monotonic() - started) * 1000,
            )
            payload = {"results": results, "coverage": coverage}
            if degraded:
                payload["degraded"] = degraded
            if warnings:
                payload["warnings"] = warnings
            if is_low_confidence(results):
                payload["low_confidence"] = True
                payload["confidence_note"] = (
                    "no lexical match and weak semantic similarity; "
                    "results are probably not relevant to the query"
                )
            return self.send_json(200, payload)
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
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
                stats = DocumentIndexer(
                    DOCUMENTS_ROOT,
                    documents_store,
                    documents_vector_store,
                    embedder.embed,
                    include_patterns=DOCUMENTS_INCLUDE_PATTERNS,
                    exclude_patterns=DOCUMENTS_EXCLUDE_PATTERNS,
                ).scan(max_files=max_files)
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
