import fcntl
import fnmatch
import hashlib
import logging
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunker import chunk_markdown
from .embeddings import hash_embedding

logger = logging.getLogger("memory_store.indexer")


@dataclass
class ScanStats:
    added: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0

    def as_dict(self):
        return asdict(self)


class Indexer:
    def __init__(
        self,
        root,
        store,
        chunk_size=1600,
        vector_store=None,
        embed=None,
        include_patterns=("**/*.md",),
        exclude_patterns=(),
    ):
        self.root, self.store, self.chunk_size, self.vector_store = (
            Path(root),
            store,
            chunk_size,
            vector_store,
        )
        self.include_patterns = tuple(include_patterns)
        self.exclude_patterns = tuple(exclude_patterns)
        self.embed = embed or (lambda text, dimensions=None: hash_embedding(text, store.dimensions))
        db_path = store.db.execute("PRAGMA database_list").fetchone()[2]
        self.lock_path = db_path + ".lock" if db_path else str(self.root / ".index.lock")

    def scan(self):
        self.store.set_index_metadata("last_index_started_at", datetime.now(timezone.utc).isoformat())
        self.store.set_index_metadata("last_index_error", "")
        if not self.root.is_dir():
            error = f"document root does not exist: {self.root}"
            self.store.set_index_metadata("last_index_error", error)
            raise FileNotFoundError(error)
        with open(self.lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                result = self._scan()
                self.store.set_index_metadata("last_index_completed_at", datetime.now(timezone.utc).isoformat())
                return result
            except Exception as exc:
                self.store.set_index_metadata("last_index_error", str(exc))
                raise

    def _scan(self):
        stats, seen = ScanStats(), set()
        candidates = {
            path
            for pattern in self.include_patterns
            for path in self.root.glob(pattern)
            if path.suffix.lower() == ".md"
        }
        for path in sorted(candidates):
            if path.is_symlink() or not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            if any(fnmatch.fnmatch(rel, pattern) for pattern in self.exclude_patterns):
                continue
            content = path.read_text(encoding="utf-8")
            seen.add(rel)
            digest, old = hashlib.sha256(content.encode()).hexdigest(), self.store.file_digest(rel)
            if old == digest:
                stats.unchanged += 1
                logger.debug("unchanged: %s", rel)
                continue
            chunks = [
                (
                    hashlib.sha256(f"{rel}\0{n}\0{c.text}".encode()).hexdigest(),
                    c.heading,
                    c.text,
                    c.start_line,
                )
                for n, c in enumerate(chunk_markdown(content, self.chunk_size))
            ]
            self.store.upsert_file(rel, digest, chunks, self.embed)
            if self.vector_store:
                if old:
                    self.vector_store.delete_file(rel)
                self.vector_store.upsert(
                    [(cid, rel, heading, body, line) for cid, heading, body, line in chunks],
                    self.embed,
                )
            if old is None:
                stats.added += 1
                logger.info("added: %s (%d chunks)", rel, len(chunks))
            else:
                stats.changed += 1
                logger.info("changed: %s (%d chunks)", rel, len(chunks))
        for rel in [r[0] for r in self.store.db.execute("SELECT path FROM files")]:
            if rel not in seen:
                self.store.delete_file(rel)
                if self.vector_store:
                    self.vector_store.delete_file(rel)
                stats.removed += 1
                logger.info("removed: %s", rel)
        return stats
