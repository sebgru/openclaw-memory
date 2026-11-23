"""Isolated mail indexing. No Graph credentials or account policy in this package."""

import argparse
import json
import os
from pathlib import Path

from .embeddings import EmbeddingClient
from .indexer import Indexer
from .qdrant import QdrantStore
from .sqlite_store import SQLiteStore


class MailQdrantStore(QdrantStore):
    """A queue item is acknowledged only after Qdrant applies each mutation."""

    def _request(self, method, path, payload=None):
        if (method == "PUT" and path.endswith("/points")) or path.endswith("/points/delete"):
            path += "?wait=true"
        return super()._request(method, path, payload)


def _prepare_vector_queue(store):
    """File changes and deletion tombstones commit atomically with SQLite state.

    Keep this mail-only: the generic Indexer's commit order is not a durable
    vector publication protocol. Also seed existing SQLite files once on upgrade.
    """
    with store.db:
        # sqlite3 does not implicitly start a transaction for DDL. Include queue
        # creation, triggers and the migration seed in one crash-safe transaction.
        store.db.execute("BEGIN IMMEDIATE")
        exists = store.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mail_vector_pending'"
        ).fetchone()
        store.db.execute("CREATE TABLE IF NOT EXISTS mail_vector_pending(path TEXT PRIMARY KEY)")
        for action, ref in (("INSERT", "NEW"), ("DELETE", "OLD")):
            store.db.execute(
                f"CREATE TRIGGER IF NOT EXISTS mail_vector_{action.lower()} AFTER {action} ON files "
                f"BEGIN INSERT OR IGNORE INTO mail_vector_pending VALUES ({ref}.path); END"
            )
        if not exists:
            store.db.execute("INSERT INTO mail_vector_pending SELECT path FROM files")


def _publish_vectors(store, vectors, embed):
    # Always replace by path. Delete/upsert are idempotent, including partial
    # batches and a crash after remote success but before the local acknowledgement.
    for (path,) in store.db.execute(
        "SELECT path FROM mail_vector_pending ORDER BY path"
    ).fetchall():
        rows = store.db.execute(
            "SELECT id, path, heading, body, line FROM chunks WHERE path=?", (path,)
        ).fetchall()
        vectors.ensure_collection()
        vectors.delete_file(path)
        vectors.upsert([tuple(row) for row in rows], embed)
        with store.db:
            store.db.execute("DELETE FROM mail_vector_pending WHERE path=?", (path,))


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
    if not (collection == "mail-private" or collection.startswith("mail-private-")):
        raise ValueError("mail collection must use the mail-private namespace")
    if collection in {
        os.getenv("QDRANT_COLLECTION", "memory"),
        os.getenv("ARCHIVE_QDRANT_COLLECTION", "memory-archive"),
        os.getenv("DOCUMENTS_QDRANT_COLLECTION", "documents"),
    }:
        raise ValueError("mail collection collides with an existing scope")
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
        _prepare_vector_queue(store)
        vectors = MailQdrantStore(qdrant_url, collection, dimensions) if qdrant_url else None
        # Resolve the generation once: an exporter publication cannot mix two snapshots.
        generation = (root / "current").resolve()
        stats = Indexer(generation / "mail", store, embed=embedder.embed).scan()
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
        if vectors:
            _publish_vectors(store, vectors, embedder.embed)
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
