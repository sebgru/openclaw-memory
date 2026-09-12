import fcntl
import hashlib
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .embeddings import EmbeddingClient
from .indexer import Indexer
from .maintenance import verify_database
from .promotion import PromotionError, candidates, promote
from .qdrant import QdrantStore
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
EXCLUDE_PATTERNS = patterns(
    "EXCLUDE_PATTERNS", DEFAULT_EXCLUDE_PATTERNS
)
ARCHIVE_ROOT = os.getenv("ARCHIVE_ROOT")
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


_DEFAULT_VECTOR = object()


def env_float(name, default):
    return float(os.getenv(name, str(default)))


def env_int(name, default):
    return int(os.getenv(name, str(default)))


PROMPT_MIN_RELEVANCE = env_float("PROMPT_MIN_RELEVANCE", 0.2)
PROMPT_SOURCE_QUOTAS = {
    "memory": env_int("PROMPT_MEMORY_QUOTA", 3),
    "artifact": env_int("PROMPT_ARTIFACT_QUOTA", 1),
    "session": env_int("PROMPT_SESSION_QUOTA", 2),
    "archive": env_int("PROMPT_ARCHIVE_QUOTA", 1),
}


def hybrid_search(query, limit, selected_store=None, selected_vector_store=_DEFAULT_VECTOR):
    selected_store = selected_store or store
    selected_vector_store = (
        vector_store if selected_vector_store is _DEFAULT_VECTOR else selected_vector_store
    )
    lexical = selected_store.search(query, limit * 3)
    try:
        semantic = (
            selected_vector_store.search(embedder.embed(query), limit * 3)
            if selected_vector_store
            else []
        )
    except Exception as exc:
        logger.warning("semantic search unavailable, falling back to FTS: %s", exc)
        semantic = []
    merged: dict = {}
    max_lexical_evidence = max((max(0.0, row.get("score", 0.0)) for row in lexical), default=0)
    for rank, row in enumerate(lexical):
        score = 1 / (60 + rank + 1)
        item = merged.setdefault(
            row["id"],
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
        item["semantic_similarity"] = max(
            item["semantic_similarity"], row.get("score", 0)
        )
    for item in merged.values():
        lexical_relevance = (
            max(0.0, item["lexical_evidence"]) / max_lexical_evidence
            if max_lexical_evidence
            else 0.0
        )
        semantic_relevance = max(0.0, min(1.0, item["semantic_similarity"]))
        item["relevance_score"] = max(lexical_relevance, semantic_relevance)
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


def vector_status(selected_store, selected_vector_store):
    return selected_vector_store.diagnostics(selected_store) if selected_vector_store else None


def annotate_result(result):
    """Attach a source label without reading the artifact itself."""
    path = str(result.get("path", ""))
    source = "artifact" if path == "outputs/INDEX.md" or path.startswith("outputs/") else "memory"
    return {**result, "source": source}


def unified_source(result, archive=False):
    """Return the stable corpus label for a unified-search result."""
    if archive:
        return "archive"
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


def unified_search(query, limit, scope="all", profile="tool"):
    """Search the configured main/archive indexes and merge partial results."""
    if scope not in {"all", "main", "archive"}:
        raise ValueError("scope must be one of all, main, archive")
    if profile not in {"tool", "prompt"}:
        raise ValueError("profile must be one of tool, prompt")
    if scope == "archive" and not archive_store:
        raise LookupError("archive is not configured")

    warnings = []
    merged: dict[str, dict] = {}
    backends = []
    if scope in {"all", "main"}:
        backends.append(("main", store, vector_store, False))
    if scope in {"all", "archive"}:
        if archive_store:
            backends.append(("archive", archive_store, archive_vector_store, True))
        else:
            warnings.append("archive: not configured")

    for name, selected_store, selected_vector_store, is_archive in backends:
        try:
            rows = hybrid_search(query, limit * 3, selected_store, selected_vector_store)
        except Exception as exc:
            logger.warning("unified %s search unavailable: %s", name, exc)
            warnings.append(f"{name}: search backend unavailable")
            continue
        for row in rows:
            source = unified_source(row, archive=is_archive)
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
    return apply_search_profile(results, limit, profile), warnings


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
        if parsed.path == "/promotion/candidates":
            return self.send_json(200, {"candidates": candidates(ROOT)})
        if parsed.path == "/archive/status":
            if not archive_store:
                return self.send_json(404, {"error": "archive is not configured"})
            integrity = verify_database(
                archive_store.db.execute("PRAGMA database_list").fetchone()[2]
            )
            status = {
                "status": integrity["status"],
                "database": integrity,
                **archive_store.status(),
            }
            if archive_vector_store:
                try:
                    status["vectors"] = vector_status(archive_store, archive_vector_store)
                except Exception as exc:
                    status["vectors"] = {"status": "error", "error": str(exc)}
            return self.send_json(200, status)
        if parsed.path in ("/search", "/archive/search"):
            is_archive = parsed.path.startswith("/archive/")
            if is_archive and not archive_store:
                return self.send_json(404, {"error": "archive is not configured"})
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
                    archive_store if is_archive else store,
                    archive_vector_store if is_archive else vector_store,
                )
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
            if scope not in {"all", "main", "archive"}:
                return self.send_json(400, {"error": "scope must be one of all, main, archive"})
            if profile not in {"tool", "prompt"}:
                return self.send_json(400, {"error": "profile must be one of tool, prompt"})
            try:
                # Keep the long-standing three-argument call shape for the
                # default profile; callers and test doubles remain compatible.
                results, warnings = (
                    unified_search(query, limit, scope)
                    if profile == "tool"
                    else unified_search(query, limit, scope, profile)
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
            payload = {"results": results}
            if warnings:
                payload["warnings"] = warnings
            return self.send_json(200, payload)
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path in ("/reconcile", "/archive/reconcile"):
            is_archive = self.path.startswith("/archive/")
            selected_store = archive_store if is_archive else store
            selected_vectors = archive_vector_store if is_archive else vector_store
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
        "starting server on port %s (document_root=%s, qdrant=%s, archive_root=%s)",
        os.getenv("PORT", "8080"),
        ROOT,
        bool(vector_store),
        ARCHIVE_ROOT or "disabled",
    )
    ThreadingHTTPServer(("0.0.0.0", int(os.getenv("PORT", "8080"))), Handler).serve_forever()
