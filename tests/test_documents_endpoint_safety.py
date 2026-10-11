import fcntl
import json
import sqlite3
import sys
import tempfile
import time
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from memory_store import server
from memory_store.document_indexer import DocumentIndexer, IndexBusyError
from memory_store.sqlite_store import SQLiteStore


class DocumentsEndpointSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.docs = root / "docs"
        self.docs.mkdir()
        (self.docs / "a.md").write_text("# A\n\nneedle\n")
        self.db_path = root / "documents.db"
        self.store = SQLiteStore(self.db_path, read_timeout=0.2)
        self.addCleanup(self.store.close)
        patches = [
            patch.object(server, "DOCUMENTS_ROOT", str(self.docs)),
            patch.object(server, "documents_store", self.store),
            patch.object(server, "documents_vector_store", None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)

    def request(self, method, path, payload=None):
        connection = HTTPConnection(*self.http.server_address, timeout=10)
        connection.request(method, path, body=None if payload is None else json.dumps(payload))
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def lock_db(self):
        writer = sqlite3.connect(str(self.db_path), isolation_level=None)
        self.addCleanup(writer.close)
        writer.execute("BEGIN EXCLUSIVE")
        return writer

    def tables(self):
        with sqlite3.connect(str(self.db_path)) as db:
            return {r[0] for r in db.execute("SELECT name FROM sqlite_master")}

    def test_status_and_errors_do_not_create_state_table(self):
        self.assertNotIn("document_file_state", self.tables())
        self.assertEqual(self.request("GET", "/documents/errors")[1], {"errors": []})
        status, body = self.request("GET", "/documents/status")
        self.assertEqual(status, 200)
        self.assertEqual(body["files_with_errors"], 0)
        self.assertNotIn("document_file_state", self.tables())

    def test_status_and_errors_return_503_promptly_when_db_locked(self):
        self.lock_db()
        for path in ("/documents/status", "/documents/errors"):
            started = time.monotonic()
            status, body = self.request("GET", path)
            self.assertEqual(status, 503, path)
            self.assertEqual(body["error"], "documents database busy")
            self.assertLess(time.monotonic() - started, 3.0, path)

    def test_read_only_indexer_does_not_write(self):
        DocumentIndexer(self.docs, self.store).scan(max_files=10)
        before = self.store.db.total_changes
        indexer = DocumentIndexer(self.docs, self.store, read_only=True)
        indexer.status()
        indexer.errors()
        self.assertEqual(self.store.db.total_changes, before)

    def test_read_only_indexer_does_not_query_shared_connection_on_init(self):
        # A stuck index operation can occupy the shared connection. Read-only
        # endpoint setup must use the filename cached when SQLiteStore opened.
        self.store.db.close()
        indexer = DocumentIndexer(self.docs, self.store, read_only=True)
        self.assertEqual(indexer.errors(), [])

    def test_index_rejects_when_scan_lock_held(self):
        indexer = DocumentIndexer(self.docs, self.store)
        with open(indexer.lock_path, "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            started = time.monotonic()
            with self.assertRaises(IndexBusyError):
                indexer.scan(max_files=10)
            self.assertLess(time.monotonic() - started, 1.0)
            started = time.monotonic()
            status, body = self.request("POST", "/documents/index", {"max_files": 10})
            self.assertEqual(status, 409)
            self.assertLess(time.monotonic() - started, 1.0)
        # a rejected request leaves index metadata untouched
        self.assertNotIn("last_index_started_at", self.store.index_metadata())
        self.assertEqual(self.request("POST", "/documents/index", {"max_files": 10})[0], 200)

    def test_lock_wait_is_bounded_and_configurable(self):
        indexer = DocumentIndexer(self.docs, self.store)
        with open(indexer.lock_path, "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            with patch.dict("os.environ", {"DOCUMENTS_INDEX_LOCK_TIMEOUT_SECONDS": "0.3"}):
                started = time.monotonic()
                with self.assertRaises(IndexBusyError):
                    indexer.scan(max_files=10)
                elapsed = time.monotonic() - started
            self.assertGreaterEqual(elapsed, 0.25)
            self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
