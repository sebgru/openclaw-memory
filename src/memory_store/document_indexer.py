import fcntl
import fnmatch
import hashlib
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .chunker import chunk_markdown

logger = logging.getLogger("memory_store.documents")


@dataclass
class DocumentScanStats:
    added: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    errors: int = 0
    pending: int = 0

    def as_dict(self):
        return asdict(self)


class DocumentIndexer:
    """Incremental, failure-isolated indexer for converted Markdown documents."""

    def __init__(
        self,
        root,
        store,
        vector_store=None,
        embed=None,
        chunk_size=1600,
        include_patterns=("**/*.md",),
        exclude_patterns=(),
    ):
        self.root = Path(root)
        self.store = store
        self.vector_store = vector_store
        self.embed = embed
        self.chunk_size = chunk_size
        self.include_patterns = tuple(include_patterns)
        self.exclude_patterns = tuple(exclude_patterns)
        db_path = store.db.execute("PRAGMA database_list").fetchone()[2]
        self.lock_path = db_path + ".lock" if db_path else str(self.root / ".documents.lock")
        self._init_state()

    def _init_state(self):
        self.store.db.execute("""
        CREATE TABLE IF NOT EXISTS document_file_state(
          path TEXT PRIMARY KEY,
          digest TEXT,
          size INTEGER,
          mtime_ns INTEGER,
          status TEXT NOT NULL,
          error TEXT,
          last_attempt_at TEXT NOT NULL,
          last_success_at TEXT
        )
        """)
        self.store.db.commit()

    def _state(
        self,
        path,
        *,
        digest=None,
        size=None,
        mtime_ns=None,
        status,
        error=None,
        success=False,
    ):
        now = datetime.now(UTC).isoformat()
        previous = self.store.db.execute(
            "SELECT last_success_at FROM document_file_state WHERE path=?", (path,)
        ).fetchone()
        last_success = now if success else (previous[0] if previous else None)
        self.store.db.execute(
            "INSERT OR REPLACE INTO document_file_state VALUES (?,?,?,?,?,?,?,?)",
            (path, digest, size, mtime_ns, status, error, now, last_success),
        )
        self.store.db.commit()

    def _successful_state(self, path):
        return self.store.db.execute(
            "SELECT digest, size, mtime_ns FROM document_file_state WHERE path=? AND status='ok'",
            (path,),
        ).fetchone()

    def _stable_read(self, path):
        for _ in range(3):
            before = path.stat()
            content = path.read_text(encoding="utf-8")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
                return content, after
        raise OSError("file changed while being read")

    def _paths(self):
        return sorted(
            {
                path
                for pattern in self.include_patterns
                for path in self.root.glob(pattern)
                if path.suffix.lower() == ".md"
                and path.is_file()
                and not path.is_symlink()
                and not any(
                    fnmatch.fnmatch(path.relative_to(self.root).as_posix(), excluded)
                    for excluded in self.exclude_patterns
                )
            }
        )

    def scan(self, max_files=250):
        if not 1 <= max_files <= 10000:
            raise ValueError("max_files must be between 1 and 10000")
        self.store.set_index_metadata("last_index_started_at", datetime.now(UTC).isoformat())
        self.store.set_index_metadata("last_index_error", "")
        if not self.root.is_dir():
            error = f"documents root does not exist: {self.root}"
            self.store.set_index_metadata("last_index_error", error)
            raise FileNotFoundError(error)
        with open(self.lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = self._scan(max_files)
        self.store.set_index_metadata("last_index_completed_at", datetime.now(UTC).isoformat())
        self.store.set_index_metadata(
            "last_index_error", "" if not result.errors else f"{result.errors} file errors"
        )
        return result

    def _scan(self, max_files):
        stats = DocumentScanStats()
        discovered = {}
        failures = {}
        for path in self._paths():
            rel = path.relative_to(self.root).as_posix()
            try:
                current = path.stat()
                previous = self._successful_state(rel)
                if (
                    previous
                    and previous[1] == current.st_size
                    and previous[2] == current.st_mtime_ns
                    and self.store.file_digest(rel) == previous[0]
                ):
                    discovered[rel] = (None, previous[0], current)
                    continue
                content, stat = self._stable_read(path)
                digest = hashlib.sha256(content.encode()).hexdigest()
                discovered[rel] = (content, digest, stat)
            except (OSError, UnicodeError) as exc:
                failures[rel] = str(exc)

        existing = {row[0] for row in self.store.db.execute("SELECT path FROM files")}
        changes = [
            rel
            for rel, (_, digest, _) in discovered.items()
            if self.store.file_digest(rel) != digest
        ]
        removals = sorted(existing - set(discovered) - set(failures))
        stats.unchanged = len(discovered) - len(changes)
        actions = [("upsert", rel) for rel in sorted(changes)] + [
            ("remove", rel) for rel in removals
        ]
        stats.pending = max(0, len(actions) - max_files)

        for rel, error in failures.items():
            self._state(rel, status="error", error=error)
            stats.errors += 1

        for action, rel in actions[:max_files]:
            try:
                if action == "remove":
                    if self.vector_store:
                        self.vector_store.delete_file(rel)
                    self.store.delete_file(rel)
                    self.store.db.execute("DELETE FROM document_file_state WHERE path=?", (rel,))
                    self.store.db.commit()
                    stats.removed += 1
                    continue
                content, digest, stat = discovered[rel]
                if content is None:  # pragma: no cover - unchanged rows are not actions
                    continue
                old_digest = self.store.file_digest(rel)
                old_ids = self.store.chunk_ids(rel)
                chunks = []
                for number, chunk in enumerate(chunk_markdown(content, self.chunk_size)):
                    chunk_id = hashlib.sha256(f"{rel}\0{number}\0{chunk.text}".encode()).hexdigest()
                    vector = self.embed(f"{chunk.heading} {chunk.text}")
                    chunks.append((chunk_id, chunk.heading, chunk.text, chunk.start_line, vector))
                if self.vector_store:
                    records = [
                        (cid, rel, heading, body, line, vector)
                        for cid, heading, body, line, vector in chunks
                    ]
                    self.vector_store.upsert_precomputed(records)
                try:
                    self.store.upsert_file_precomputed(rel, digest, chunks)
                except Exception:
                    if self.vector_store:
                        self.vector_store.delete_ids(
                            [row[0] for row in chunks if row[0] not in set(old_ids)]
                        )
                    raise
                if self.vector_store:
                    new_ids = {row[0] for row in chunks}
                    self.vector_store.delete_ids([item for item in old_ids if item not in new_ids])
                self._state(
                    rel,
                    digest=digest,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    status="ok",
                    success=True,
                )
                if old_digest is None:
                    stats.added += 1
                else:
                    stats.changed += 1
            except Exception as exc:
                logger.warning("document indexing failed for %s: %s", rel, exc)
                content_info = discovered.get(rel)
                self._state(
                    rel,
                    digest=content_info[1] if content_info else None,
                    size=content_info[2].st_size if content_info else None,
                    mtime_ns=content_info[2].st_mtime_ns if content_info else None,
                    status="error",
                    error=str(exc),
                )
                stats.errors += 1
        return stats

    def status(self):
        result = self.store.status()
        result["files_with_errors"] = self.store.db.execute(
            "SELECT count(*) FROM document_file_state WHERE status='error'"
        ).fetchone()[0]
        result["pending_hint"] = "run another bounded index batch until pending is zero"
        return result
