"""Isolated mail indexing. No Graph credentials or account policy in this package."""

import argparse
import json
import os
from pathlib import Path

from .embeddings import EmbeddingClient
from .indexer import Indexer
from .qdrant import QdrantStore
from .sqlite_store import SQLiteStore


def _build(
    export_root: str,
    database: str,
    dimensions: int = 128,
    qdrant_url: str | None = None,
    collection: str = "mail-private",
) -> dict:
    """Incrementally reconcile a completed export into an explicitly separate SQLite DB."""
    root = Path(export_root).resolve()
    if not (root / "current" / "state.json").is_file():
        raise ValueError("completed mail export required")
    if not Path(database).is_absolute():
        raise ValueError("mail database path must be absolute")
    target = Path(database).resolve()
    if not target.is_absolute() or target.name in {"memory.db", "documents.db", "archive.db"}:
        raise ValueError("mail database must use a distinct explicit path")
    for name in ("SQLITE_PATH", "DOCUMENTS_SQLITE_PATH", "ARCHIVE_SQLITE_PATH"):
        if os.environ.get(name) and target == Path(os.environ[name]).resolve():
            raise ValueError("mail database collides with an existing scope")
    if not collection.startswith("mail-"):
        raise ValueError("mail collection must use the mail- namespace")
    embedder = EmbeddingClient(
        os.getenv("EMBEDDING_URL"), os.getenv("EMBEDDING_MODEL", "default"), dimensions
    )
    target.touch(mode=0o600, exist_ok=True)
    target.chmod(0o600)
    store = SQLiteStore(str(target), dimensions)
    try:
        # Mailboxes are much larger than curated memory. Avoid quadratic path scans
        # during each upsert/delete while retaining the existing SQLiteStore schema.
        store.db.execute("CREATE INDEX IF NOT EXISTS idx_mail_chunks_path ON chunks(path)")
        store.db.commit()
        vectors = QdrantStore(qdrant_url, collection, dimensions) if qdrant_url else None
        # Resolve the generation once: an exporter publication cannot mix two snapshots.
        generation = (root / "current").resolve()
        stats = Indexer(
            generation / "mail", store, embed=embedder.embed, vector_store=vectors
        ).scan()
        state = json.loads((generation / "state.json").read_text())
        with store.db:
            store.db.execute(
                "CREATE TABLE IF NOT EXISTS mail_metadata("
                "path TEXT PRIMARY KEY, subject TEXT, sender TEXT, date TEXT)"
            )
            store.db.execute("DELETE FROM mail_metadata")
            for message_id, filename in state.get("aliases", {}).items():
                message = state["records"][message_id]
                sender = ((message.get("from") or {}).get("emailAddress") or {}).get("address", "")
                store.db.execute(
                    "INSERT OR IGNORE INTO mail_metadata VALUES (?,?,?,?)",
                    (
                        filename,
                        message.get("subject", ""),
                        sender,
                        message.get("receivedDateTime", ""),
                    ),
                )
        return stats.as_dict()
    finally:
        store.db.close()


def build(
    export_root: str,
    database: str,
    dimensions: int = 128,
    qdrant_url: str | None = None,
    collection: str = "mail-private",
) -> dict:
    """Serialize publication and indexing so removed generations cannot disappear mid-scan."""
    root = Path(export_root).resolve()
    if not (root / "current" / "state.json").is_file():
        raise ValueError("completed mail export required")
    lock = root / ".sync.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        return _build(str(root), database, dimensions, qdrant_url, collection)
    finally:
        os.close(descriptor)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--qdrant-url")
    parser.add_argument("--collection", default="mail-private")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.export_root, args.database, args.dimensions, args.qdrant_url, args.collection
            )
        )
    )


if __name__ == "__main__":
    main()
