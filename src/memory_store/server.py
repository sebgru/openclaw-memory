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
EXCLUDE_PATTERNS = patterns("EXCLUDE_PATTERNS", "**/review-candidates/**,**/archive/**")
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
    for rank, row in enumerate(lexical):
        score = 1 / (60 + rank + 1)
        item = merged.setdefault(
            row["id"], {**row, "score": 0, "lexical_score": 0, "semantic_score": 0}
        )
        item["score"] += score
        item["lexical_score"] += score
    for rank, row in enumerate(semantic):
        score = 1 / (60 + rank + 1)
        item = merged.setdefault(
            row["id"],
            {
                **row,
                "score": row.get("score", 0),
                "lexical_score": 0,
                "semantic_score": 0,
            },
        )
        item["score"] += score
        item["semantic_score"] += score
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:limit]


def indexer(root, selected_store, selected_vector_store, includes=("**/*.md",), excludes=()):
    return Indexer(
        root,
        selected_store,
        vector_store=selected_vector_store,
        embed=embedder.embed,
        include_patterns=includes,
        exclude_patterns=excludes,
    )


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
            return self.send_json(
                200,
                {
                    "status": integrity["status"],
                    "backend": "sqlite+qdrant" if vector_store else "sqlite",
                    "database": integrity,
                    **store.status(),
                },
            )
        if parsed.path == "/promotion/candidates":
            return self.send_json(200, {"candidates": candidates(ROOT)})
        if parsed.path == "/archive/status":
            if not archive_store:
                return self.send_json(404, {"error": "archive is not configured"})
            return self.send_json(200, {"status": "ok", **archive_store.status()})
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
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
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
