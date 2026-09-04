import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .embeddings import EmbeddingClient
from .indexer import Indexer
from .qdrant import QdrantStore
from .sqlite_store import SQLiteStore

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
    except Exception:
        semantic = []
    merged: dict = {}
    for rank, row in enumerate(lexical):
        merged.setdefault(row["id"], {**row, "score": 0})["score"] += 1 / (60 + rank + 1)
    for rank, row in enumerate(semantic):
        merged.setdefault(row["id"], row)["score"] += 1 / (60 + rank + 1)
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
        parsed, params = urlparse(self.path), parse_qs(urlparse(self.path).query)
        if parsed.path in ("/healthz", "/status"):
            return self.send_json(
                200,
                {
                    "status": "ok",
                    "backend": "sqlite+qdrant" if vector_store else "sqlite",
                    **store.status(),
                },
            )
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
                return self.send_json(
                    200,
                    {
                        "results": hybrid_search(
                            query,
                            limit,
                            archive_store if is_archive else store,
                            archive_vector_store if is_archive else vector_store,
                        )
                    },
                )
            except Exception as exc:
                return self.send_json(
                    503, {"error": "search backend unavailable", "detail": str(exc)}
                )
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/index", "/archive/index"):
            return self.send_json(404, {"error": "not found"})
        is_archive = self.path == "/archive/index"
        if is_archive and not archive_store:
            return self.send_json(404, {"error": "archive is not configured"})
        try:
            return self.send_json(
                200,
                indexer(
                    ARCHIVE_ROOT if is_archive else ROOT,
                    archive_store if is_archive else store,
                    archive_vector_store if is_archive else vector_store,
                    ("**/*.md",) if is_archive else INCLUDE_PATTERNS,
                    () if is_archive else EXCLUDE_PATTERNS,
                )
                .scan()
                .as_dict(),
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return self.send_json(400, {"error": str(exc)})

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(os.getenv("PORT", "8080"))), Handler).serve_forever()
