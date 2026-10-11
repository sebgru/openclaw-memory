import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from memory_store import server
from memory_store.sqlite_store import SQLiteStore


def seed(store, path="a.md", text="lockprobe content"):
    store.upsert_file(path, "d", [("c1", "H", text, 1)])


class ReadIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "docs.db"

    def test_read_timeout_env_is_honored_and_clamped(self):
        with patch.dict(os.environ, {"SQLITE_READ_TIMEOUT_SECONDS": "0.5"}):
            store = SQLiteStore(":memory:")
        self.assertEqual(store.read_timeout, 0.5)
        with patch.dict(os.environ, {"SQLITE_READ_TIMEOUT_SECONDS": "-3"}):
            self.assertEqual(SQLiteStore(":memory:").read_timeout, 0.0)
        self.assertEqual(SQLiteStore(":memory:", read_timeout=3.0).read_timeout, 3.0)

    def test_invalid_read_timeout_env_falls_back_to_default(self):
        with patch.dict(os.environ, {"SQLITE_READ_TIMEOUT_SECONDS": "not-a-number"}):
            store = SQLiteStore(":memory:")
        self.assertEqual(store.read_timeout, 2.0)

    def test_memory_store_keeps_shared_connection(self):
        store = SQLiteStore(":memory:")
        seed(store)
        with store._read_connection() as db:
            self.assertIs(db, store.db)
        self.assertEqual(len(store.search("lockprobe")), 1)

    def test_file_store_search_works_and_sees_committed_rows(self):
        store = SQLiteStore(self.path, read_timeout=0.2)
        seed(store)
        self.assertEqual(store.search("lockprobe")[0]["id"], "c1")

    def test_search_fails_fast_while_writer_holds_exclusive_lock(self):
        store = SQLiteStore(self.path, read_timeout=0.2)
        seed(store)
        writer = sqlite3.connect(str(self.path), isolation_level=None)
        self.addCleanup(writer.close)
        writer.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        with self.assertRaises(sqlite3.OperationalError):
            store.search("lockprobe")
        self.assertLess(time.monotonic() - started, 2.0)
        writer.execute("ROLLBACK")
        self.assertEqual(len(store.search("lockprobe")), 1)

    def test_search_not_blocked_by_open_write_transaction(self):
        store = SQLiteStore(self.path, read_timeout=0.2)
        seed(store)
        with store.write_lock:
            store.db.execute("BEGIN")
            store.db.execute("DELETE FROM files WHERE path='a.md'")
            started = time.monotonic()
            self.assertEqual(len(store.search("lockprobe")), 1)
            self.assertLess(time.monotonic() - started, 2.0)
            store.db.rollback()

    def test_unified_all_scope_keeps_main_and_archive_when_documents_locked(self):
        main = SQLiteStore(self.path.with_name("main.db"))
        archive = SQLiteStore(self.path.with_name("archive.db"))
        docs = SQLiteStore(self.path, read_timeout=0.2)
        seed(main, "m.md", "lockprobe main")
        seed(archive, "x.md", "lockprobe archive")
        seed(docs, "d.md", "lockprobe doc")
        writer = sqlite3.connect(str(self.path), isolation_level=None)
        self.addCleanup(writer.close)
        writer.execute("BEGIN EXCLUSIVE")
        with (
            patch.object(server, "store", main),
            patch.object(server, "vector_store", None),
            patch.object(server, "archive_store", archive),
            patch.object(server, "archive_vector_store", None),
            patch.object(server, "documents_store", docs),
            patch.object(server, "documents_vector_store", None),
        ):
            started = time.monotonic()
            results, warnings, coverage, _ = server.unified_search_with_coverage(
                "lockprobe", 10, "all"
            )
        self.assertLess(time.monotonic() - started, 3.0)
        sources = {r["source"] for r in results}
        self.assertEqual(sources, {"memory", "archive"})
        self.assertIn("documents: search backend unavailable", warnings)
        self.assertEqual(
            coverage, {"main": "searched", "archive": "searched", "documents": "unavailable"}
        )


if __name__ == "__main__":
    unittest.main()
