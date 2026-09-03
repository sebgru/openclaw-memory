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
embedder = EmbeddingClient(os.getenv("EMBEDDING_URL"), os.getenv("EMBEDDING_MODEL", "default"), DIM)
store = SQLiteStore(os.getenv("SQLITE_PATH", "memory.db"), DIM)
vector_store = (
    QdrantStore(os.getenv("QDRANT_URL"), os.getenv("QDRANT_COLLECTION", "memory"), DIM)
    if os.getenv("QDRANT_URL")
    else None
)


def hybrid_search(query, limit):
    lexical = store.search(query, limit * 3)
    semantic = vector_store.search(embedder.embed(query), limit * 3) if vector_store else []
    merged: dict = {}
    for rank, row in enumerate(lexical):
        merged.setdefault(row["id"], {**row, "score": 0})["score"] += 1 / (60 + rank + 1)
    for rank, row in enumerate(semantic):
        merged.setdefault(row["id"], row)["score"] += 1 / (60 + rank + 1)
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:limit]


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
        if parsed.path == "/search":
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
                return self.send_json(200, {"results": hybrid_search(query, limit)})
            except Exception as exc:
                return self.send_json(
                    503, {"error": "search backend unavailable", "detail": str(exc)}
                )
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/index":
            return self.send_json(404, {"error": "not found"})
        try:
            return self.send_json(
                200,
                Indexer(ROOT, store, vector_store=vector_store, embed=embedder.embed)
                .scan()
                .as_dict(),
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return self.send_json(400, {"error": str(exc)})

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(os.getenv("PORT", "8080"))), Handler).serve_forever()
