import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .embeddings import hash_embedding


class SQLiteStore:
    def __init__(self, path: str | Path = ":memory:", dimensions: int = 128):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.dimensions = dimensions
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, digest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, path TEXT NOT NULL, heading TEXT, body TEXT NOT NULL, line INTEGER, vector BLOB);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, path, heading, body);
        CREATE TABLE IF NOT EXISTS index_metadata(key TEXT PRIMARY KEY, value TEXT);
        """)
        self.db.commit()

    def file_digest(self, path):
        row = self.db.execute("SELECT digest FROM files WHERE path=?", (path,)).fetchone()
        return row[0] if row else None

    def set_index_metadata(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO index_metadata VALUES (?, ?)", (key, value))
        self.db.commit()

    def index_metadata(self):
        return {row[0]: row[1] for row in self.db.execute("SELECT key, value FROM index_metadata")}

    def upsert_file(self, path, digest, chunks, embed=None):
        self.delete_file(path, commit=False)
        self.db.execute("INSERT INTO files VALUES (?,?)", (path, digest))
        for chunk_id, heading, body, line in chunks:
            vector = (embed or (lambda text: hash_embedding(text, self.dimensions)))(
                f"{heading} {body}"
            )
            vec = ",".join(map(str, vector))
            self.db.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                (chunk_id, path, heading, body, line, vec),
            )
            self.db.execute(
                "INSERT INTO chunks_fts VALUES (?,?,?,?)", (chunk_id, path, heading, body)
            )
        self.db.commit()

    def delete_file(self, path, commit=True):
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE path=?", (path,))]
        self.db.execute("DELETE FROM files WHERE path=?", (path,))
        self.db.execute("DELETE FROM chunks WHERE path=?", (path,))
        for item in ids:
            self.db.execute("DELETE FROM chunks_fts WHERE id=?", (item,))
        if commit:
            self.db.commit()

    def search(self, query, limit=10):
        if not query.strip():
            return []
        safe = " OR ".join('"' + x.replace('"', "") + '"' for x in query.split() if x)
        rows = self.db.execute(
            "SELECT c.*, bm25(chunks_fts) AS rank FROM chunks_fts f JOIN chunks c ON c.id=f.id WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (safe, limit),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "path": r["path"],
                "heading": r["heading"],
                "text": r["body"],
                "line": r["line"],
                "score": float(-r["rank"]),
            }
            for r in rows
        ]

    def close(self):
        self.db.close()

    def vector(self, chunk_id):
        row = self.db.execute("SELECT vector FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return [float(x) for x in row[0].split(",")] if row else None

    def status(self):
        metadata = self.index_metadata()
        age = None
        if metadata.get("last_index_completed_at"):
            try:
                age = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(metadata["last_index_completed_at"])).total_seconds())
            except ValueError:
                pass
        return {
            "files": self.db.execute("SELECT count(*) FROM files").fetchone()[0],
            "chunks": self.db.execute("SELECT count(*) FROM chunks").fetchone()[0],
            "index": {**metadata, "age_seconds": age},
        }
