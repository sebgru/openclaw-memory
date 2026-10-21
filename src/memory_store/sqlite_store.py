import os
import sqlite3
import threading
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .embeddings import hash_embedding


def _default_read_timeout() -> float:
    try:
        return max(0.0, float(os.getenv("SQLITE_READ_TIMEOUT_SECONDS", "2")))
    except ValueError:
        return 2.0



FUZZY_MAX_CANDIDATES = 20
FUZZY_MAX_TOKENS = 4
FUZZY_SCAN_LIMIT = 200


def _fold(text):
    """Casefold and strip diacritics (ß -> ss) for typo comparison."""
    import unicodedata

    text = text.casefold().replace("ß", "ss")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _is_typo_variant(token, word):
    """True when *word* is a one-edit typo of the already folded *token*.

    A word that only differs by folding (ß/ss, umlauts) also counts. The edit
    must be interior: one leading character and two trailing characters have to
    match, which rejects German inflections (Rechnung/Rechnungen, Projekt/
    Projekte, Termin/Termine, Strom/Stroms, kauft/kaufe) while accepting
    fussmatte/fyssmatte. Exact (raw) matches are lexical evidence, not typos.
    """
    if word.lower() == token:
        return False
    word = _fold(word)
    if word == token:
        return True
    if abs(len(word) - len(token)) > 1:
        return False
    prefix = 0
    limit = min(len(word), len(token))
    while prefix < limit and word[prefix] == token[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and word[-1 - suffix] == token[-1 - suffix]:
        suffix += 1
    if prefix < 1 or suffix < 2:
        return False
    mid_w = word[prefix:len(word) - suffix]
    mid_t = token[prefix:len(token) - suffix]
    if len(mid_w) <= 1 and len(mid_t) <= 1:
        return True  # substitution, insertion or deletion
    return len(mid_w) == len(mid_t) == 2 and mid_w == mid_t[::-1]  # transposition


class SQLiteStore:
    def __init__(
        self,
        path: str | Path = ":memory:",
        dimensions: int = 128,
        read_timeout: float | None = None,
        enable_fuzzy: bool = True,
    ):
        self.path = str(path)
        self.enable_fuzzy = enable_fuzzy
        # Upper bound on how long a search waits for a file lock held by a
        # long-running writer before failing fast with ``OperationalError``.
        self.read_timeout = _default_read_timeout() if read_timeout is None else read_timeout
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        # Cache the database filename while the store is being initialized.
        # Read-only document handlers must not query the shared connection just
        # to discover the lock-file path: that connection may be occupied by
        # a long-running index transaction.
        self.database_path = self.db.execute("PRAGMA database_list").fetchone()[2]
        self.dimensions = dimensions
        # Re-entrant lock guarding every write on this shared connection.
        # A multi-statement transaction (``upsert_file_precomputed``) must not
        # be overtaken by another writer committing the same connection, which
        # surfaced as ``cannot commit - no transaction is active``.
        self._write_lock = threading.RLock()
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, digest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, path TEXT NOT NULL, heading TEXT, body TEXT NOT NULL, line INTEGER, vector BLOB);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, path, heading, body);
        CREATE TABLE IF NOT EXISTS index_metadata(key TEXT PRIMARY KEY, value TEXT);
        """)
        if self.enable_fuzzy:
            self._ensure_trigram_index()
        self.db.commit()

    def _ensure_trigram_index(self):
        """Create and one-time backfill the trigram index.

        ``chunks_trigram.id`` is UNINDEXED, so a per-row ``NOT EXISTS`` probe
        would scan the whole FTS table for every chunk (O(n^2)). Instead a
        marker row, written in the same transaction as the bulk copy, makes the
        migration atomic and idempotent: startup after the first one is O(1).
        """
        with self._write_lock, self.db:
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_trigram USING fts5("
                "id UNINDEXED, path, heading, body, tokenize='trigram')"
            )
            self.db.execute("CREATE TABLE IF NOT EXISTS trigram_state(key TEXT PRIMARY KEY)")
            if self.db.execute(
                "SELECT 1 FROM trigram_state WHERE key='backfilled'"
            ).fetchone():
                return
            self.db.execute("DELETE FROM chunks_trigram")
            self.db.execute(
                "INSERT INTO chunks_trigram (id, path, heading, body) "
                "SELECT id, path, heading, body FROM chunks"
            )
            self.db.execute("INSERT INTO trigram_state VALUES ('backfilled')")

    @property
    def write_lock(self):
        """Lock that serialises every write to this store's shared connection.

        Callers performing their own multi-statement update (for example the
        document indexer's file-state table) hold this lock so their writes
        cannot interleave with an in-flight store transaction.
        """
        return self._write_lock

    def file_digest(self, path):
        row = self.db.execute("SELECT digest FROM files WHERE path=?", (path,)).fetchone()
        return row[0] if row else None

    def set_index_metadata(self, key, value):
        with self._write_lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO index_metadata VALUES (?, ?)", (key, value))

    def index_metadata(self, db=None):
        return {
            row[0]: row[1]
            for row in (db or self.db).execute("SELECT key, value FROM index_metadata")
        }

    def upsert_file(self, path, digest, chunks, embed=None):
        with self._write_lock, self.db:
            self._delete_file_no_commit(path)
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
                if self.enable_fuzzy:
                    self.db.execute(
                        "INSERT INTO chunks_trigram VALUES (?,?,?,?)",
                        (chunk_id, path, heading, body),
                    )

    def upsert_file_precomputed(self, path, digest, chunks):
        """Atomically replace a file using already-computed vectors.

        The whole replacement runs under ``self._write_lock`` and commits via
        the sqlite3 connection context manager, so no other writer can commit
        this connection mid-transaction.  The earlier explicit
        ``BEGIN``/``COMMIT`` form could surface ``cannot commit - no
        transaction is active`` when a concurrent request committed the shared
        connection between the last insert and the commit.
        """
        with self._write_lock, self.db:
            self._delete_file_no_commit(path)
            self.db.execute("INSERT INTO files VALUES (?,?)", (path, digest))
            for chunk_id, heading, body, line, vector in chunks:
                vec = ",".join(map(str, vector))
                self.db.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                    (chunk_id, path, heading, body, line, vec),
                )
                self.db.execute(
                    "INSERT INTO chunks_fts VALUES (?,?,?,?)",
                    (chunk_id, path, heading, body),
                )
                if self.enable_fuzzy:
                    self.db.execute(
                        "INSERT INTO chunks_trigram VALUES (?,?,?,?)",
                        (chunk_id, path, heading, body),
                    )

    def _delete_file_no_commit(self, path):
        """Delete *path*'s rows; the caller owns the surrounding transaction."""
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE path=?", (path,))]
        self.db.execute("DELETE FROM files WHERE path=?", (path,))
        self.db.execute("DELETE FROM chunks WHERE path=?", (path,))
        for item in ids:
            self.db.execute("DELETE FROM chunks_fts WHERE id=?", (item,))
            if self.enable_fuzzy:
                self.db.execute("DELETE FROM chunks_trigram WHERE id=?", (item,))

    def chunk_ids(self, path):
        return [row[0] for row in self.db.execute("SELECT id FROM chunks WHERE path=?", (path,))]

    def delete_file(self, path, commit=True):
        """Delete *path* under the write lock.

        ``commit=False`` is used by callers that already hold the lock and own
        the surrounding transaction; committing here would break their
        atomicity.
        """
        with self._write_lock:
            self._delete_file_no_commit(path)
            if commit:
                self.db.commit()

    @contextmanager
    def _read_connection(self):
        """Yield a connection for read-only queries.

        File-backed stores get a short-lived read-only connection with a
        bounded busy timeout, so a search neither queues behind the shared
        connection's mutex while a long index transaction runs nor waits
        indefinitely for a file lock.  ``:memory:`` databases exist only on
        the shared connection, which is reused unchanged.  Writes keep going
        through the single shared connection and ``write_lock``.
        """
        if self.path == ":memory:" or self.path.startswith("file:"):
            yield self.db
            return
        uri = Path(self.path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=self.read_timeout)) as db:
            db.row_factory = sqlite3.Row
            yield db

    def search(self, query, limit=10):
        if not query.strip():
            return []
        safe = " OR ".join('"' + x.replace('"', "") + '"' for x in query.split() if x)
        with self._read_connection() as db:
            rows = db.execute(
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

    def search_exact(self, query, limit=32):
        """Fetch a small FTS candidate set containing every significant query term.

        This supplements the broad OR search so a distinctive model/name match
        cannot be crowded out by common-term hits before ranking. FTS5 performs
        indexed token lookups; result count is capped to keep work bounded.
        """
        terms = [term.strip('"') for term in query.split() if len(term.strip('"')) >= 3]
        if not terms:
            return []
        match = " AND ".join('"' + term.replace('"', "") + '"' for term in terms)
        limit = max(1, min(int(limit), 100))
        with self._read_connection() as db:
            rows = db.execute(
                "SELECT c.*, bm25(chunks_fts) AS rank FROM chunks_fts f "
                "JOIN chunks c ON c.id=f.id WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "path": row["path"],
                "heading": row["heading"],
                "text": row["body"],
                "line": row["line"],
                "score": float(-row["rank"]),
            }
            for row in rows
        ]

    def close(self):
        self.db.close()

    def search_fuzzy(self, query, limit=20):
        """Return typo-tolerant candidates for distinctive query tokens.

        FTS5's trigram index narrows the corpus; each candidate word is then
        checked with ``_is_typo_variant`` (single edit, not an inflection).
        Candidates are bounded by ``FUZZY_MAX_CANDIDATES`` regardless of the
        caller's limit and returned best-first; ``score`` is only a stable
        ordering key, not lexical evidence.
        """
        if not self.enable_fuzzy:
            return []
        import re

        limit = min(limit, FUZZY_MAX_CANDIDATES)
        tokens = sorted(
            {_fold(t) for t in re.findall(r"[\w-]+", query) if 5 <= len(t) <= 64},
            key=len,
            reverse=True,
        )[:FUZZY_MAX_TOKENS]
        if not tokens:
            return []
        trigrams = sorted({token[i:i + 3] for token in tokens for i in range(len(token) - 2)})
        match = " OR ".join('"' + tri.replace('"', '""') + '"' for tri in trigrams)
        with self._read_connection() as db:
            rows = db.execute(
                "SELECT c.* FROM chunks_trigram t JOIN chunks c ON c.id=t.id "
                "WHERE chunks_trigram MATCH ? ORDER BY bm25(chunks_trigram) LIMIT ?",
                (match, FUZZY_SCAN_LIMIT),
            ).fetchall()
        results = []
        for row in rows:
            words = re.findall(r"[\w-]+", f"{row['heading'] or ''} {row['body']} {row['path']}")
            hit = any(
                _is_typo_variant(token, word)
                for token in tokens
                for word in words
                if len(word) >= 5
            )
            if hit:
                results.append(
                    {
                        "id": row["id"],
                        "path": row["path"],
                        "heading": row["heading"],
                        "text": row["body"],
                        "line": row["line"],
                        "score": 1.0 - len(results) / (FUZZY_SCAN_LIMIT + 1),
                        "fuzzy_match": True,
                    }
                )
                if len(results) >= limit:
                    break
        return results

    def vector(self, chunk_id):
        row = self.db.execute("SELECT vector FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return [float(x) for x in row[0].split(",")] if row else None

    def status(self, db=None, include_fts_diagnostics=True, quick=False):
        """Return index counts; pass a ``_read_connection()`` handle to avoid the shared one."""
        db = db or self.db
        metadata = self.index_metadata(db)
        chunk_count = None if quick else db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        if include_fts_diagnostics and not quick:
            fts_count = db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
            missing_fts = db.execute(
                "SELECT count(*) FROM chunks c LEFT JOIN chunks_fts f ON f.id=c.id WHERE f.id IS NULL"
            ).fetchone()[0]
            orphaned_fts = db.execute(
                "SELECT count(*) FROM chunks_fts f LEFT JOIN chunks c ON c.id=f.id WHERE c.id IS NULL"
            ).fetchone()[0]
            lexical_index = {
                "chunks": fts_count,
                "missing": missing_fts,
                "orphaned": orphaned_fts,
                "inSync": (fts_count == chunk_count and missing_fts == 0 and orphaned_fts == 0),
            }
        else:
            # Avoid full-table FTS parity joins on every health/status poll.
            lexical_index = {
                "chunks": None,
                "missing": None,
                "orphaned": None,
                "inSync": None,
                "check": "not_run",
            }
        age = None
        if metadata.get("last_index_completed_at"):
            try:
                age = max(
                    0.0,
                    (
                        datetime.now(UTC)
                        - datetime.fromisoformat(metadata["last_index_completed_at"])
                    ).total_seconds(),
                )
            except ValueError:
                pass
        return {
            # File count stays exact: it is a much smaller metadata table than
            # chunks and useful for backward-compatible status reporting.
            "files": db.execute("SELECT count(*) FROM files").fetchone()[0],
            "chunks": chunk_count,
            "lexicalIndex": lexical_index,
            "index": {**metadata, "age_seconds": age},
        }
