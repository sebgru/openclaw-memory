import json
import sys
import tempfile
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from memory_store import server
from memory_store.document_indexer import DocumentIndexer
from memory_store.sqlite_store import SQLiteStore


class FakeVectors:
    def __init__(self):
        self.rows = {}
        self.fail = False

    def upsert_precomputed(self, records):
        if self.fail:
            raise OSError("vector backend down")
        for row in records:
            self.rows[row[0]] = row

    def delete_ids(self, ids):
        for item in ids:
            self.rows.pop(item, None)

    def delete_file(self, path):
        self.rows = {key: row for key, row in self.rows.items() if row[1] != path}

    def diagnostics(self, store):
        chunks = store.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return {
            "points": len(self.rows),
            "chunks": chunks,
            "missing_vectors": chunks - len(self.rows),
        }

    def reconcile_missing(self, store, embed, limit):
        return {"reconciled": 0, "remaining": 0}


class DocumentIndexerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "converted"
        self.root.mkdir()
        self.store = SQLiteStore(Path(self.tmp.name) / "documents.db", 8)
        self.vectors = FakeVectors()
        self.indexer = DocumentIndexer(
            self.root,
            self.store,
            self.vectors,
            lambda text: [0.125] * 8,
        )

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_incremental_lifecycle_and_bounded_resume(self):
        (self.root / "a.md").write_text("# A\nalpha")
        (self.root / "b.md").write_text("# B\nbeta")
        first = self.indexer.scan(max_files=1)
        self.assertEqual((first.added, first.pending), (1, 1))
        second = self.indexer.scan(max_files=1)
        self.assertEqual((second.added, second.pending), (1, 0))
        (self.root / "a.md").write_text("# A\nalpha changed")
        changed = self.indexer.scan()
        self.assertEqual(changed.changed, 1)
        (self.root / "b.md").rename(self.root / "renamed.md")
        renamed = self.indexer.scan()
        self.assertEqual((renamed.added, renamed.removed), (1, 1))
        (self.root / "a.md").unlink()
        removed = self.indexer.scan()
        self.assertEqual(removed.removed, 1)

    def test_failure_isolated_and_last_valid_version_remains(self):
        path = self.root / "good.md"
        path.write_text("# Good\nold searchable text")
        self.indexer.scan()
        old_digest = self.store.file_digest("good.md")
        path.write_text("# Good\nnew text")
        self.vectors.fail = True
        stats = self.indexer.scan()
        self.assertEqual(stats.errors, 1)
        self.assertEqual(self.store.file_digest("good.md"), old_digest)
        self.assertEqual(self.store.search("old", 1)[0]["path"], "good.md")

    def test_invalid_utf8_does_not_block_other_files(self):
        (self.root / "bad.md").write_bytes(b"\xff")
        (self.root / "good.md").write_text("good")
        result = self.indexer.scan()
        self.assertEqual((result.added, result.errors), (1, 1))
        self.assertEqual(self.indexer.status()["files_with_errors"], 1)

    def test_delete_failure_retains_last_valid_searchable_version(self):
        path = self.root / "keep.md"
        path.write_text("retain this")
        self.indexer.scan()
        path.unlink()
        self.vectors.fail = True
        original = self.vectors.delete_file
        self.vectors.delete_file = lambda path: (_ for _ in ()).throw(OSError("down"))
        try:
            result = self.indexer.scan()
        finally:
            self.vectors.delete_file = original
        self.assertEqual(result.errors, 1)
        self.assertIsNotNone(self.store.file_digest("keep.md"))
        self.assertEqual(self.store.search("retain", 1)[0]["path"], "keep.md")

    def test_existing_database_is_migrated_with_document_state(self):
        tables = {
            row[0]
            for row in self.store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        self.assertIn("document_file_state", tables)

    def test_missing_root_records_error(self):
        missing = DocumentIndexer(
            self.root / "missing",
            self.store,
            self.vectors,
            lambda text: [0.125] * 8,
        )
        with self.assertRaises(FileNotFoundError):
            missing.scan()
        self.assertIn("does not exist", self.store.index_metadata()["last_index_error"])

    def test_fts_only_deletion(self):
        indexer = DocumentIndexer(
            self.root,
            self.store,
            None,
            lambda text: [0.125] * 8,
        )
        path = self.root / "delete.md"
        path.write_text("delete me")
        indexer.scan()
        path.unlink()
        self.assertEqual(indexer.scan().removed, 1)

    def test_sqlite_failure_removes_only_new_vectors(self):
        path = self.root / "rollback.md"
        path.write_text("old body")
        self.indexer.scan()
        old_ids = set(self.store.chunk_ids("rollback.md"))
        path.write_text("entirely new body")
        original = self.store.upsert_file_precomputed

        def fail(*args):
            raise OSError("sqlite full")

        self.store.upsert_file_precomputed = fail
        try:
            result = self.indexer.scan()
        finally:
            self.store.upsert_file_precomputed = original
        self.assertEqual(result.errors, 1)
        self.assertEqual(set(self.vectors.rows), old_ids)


class DocumentApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "converted"
        cls.root.mkdir()
        (cls.root / "manual.md").write_text("# Manual\ndocument needle")
        cls.store = SQLiteStore(Path(cls.tmp.name) / "documents.db", 8)
        cls.previous = (
            server.DOCUMENTS_ROOT,
            server.documents_store,
            server.documents_vector_store,
            server.DOCUMENTS_INCLUDE_PATTERNS,
        )
        server.DOCUMENTS_ROOT = str(cls.root)
        server.documents_store = cls.store
        server.documents_vector_store = None
        server.DOCUMENTS_INCLUDE_PATTERNS = ("**/*.md",)
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        Thread(target=cls.http.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.store.close()
        (
            server.DOCUMENTS_ROOT,
            server.documents_store,
            server.documents_vector_store,
            server.DOCUMENTS_INCLUDE_PATTERNS,
        ) = cls.previous
        cls.tmp.cleanup()

    def request(self, method, path, payload=None):
        connection = HTTPConnection(*self.http.server_address)
        body = None if payload is None else json.dumps(payload)
        connection.request(method, path, body=body)
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def test_index_status_search_and_fts_fallback(self):
        status, body = self.request("POST", "/documents/index", {"max_files": 10})
        self.assertEqual(status, 200)
        self.assertEqual(body["added"] + body["unchanged"], 1)
        self.assertEqual(self.request("GET", "/documents/status")[0], 200)
        status, body = self.request("GET", "/documents/search?q=needle")
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["source"], "document")

    def test_index_validation(self):
        self.assertEqual(self.request("POST", "/documents/index", {"max_files": 0})[0], 400)

    def test_unified_document_scope_and_federation(self):
        status, body = self.request("GET", "/unified/search?q=needle&scope=documents")
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["source"], "document")

    def test_document_semantic_failure_falls_back_to_fts(self):
        class FailingVectors:
            def search(self, vector, limit):
                raise OSError("qdrant down")

        self.request("POST", "/documents/index", {"max_files": 10})
        original = server.documents_vector_store
        server.documents_vector_store = FailingVectors()
        try:
            status, body = self.request("GET", "/documents/search?q=needle")
        finally:
            server.documents_vector_store = original
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["source"], "document")

    def test_unified_document_failure_returns_partial_warning(self):
        original = server.hybrid_search
        try:

            def failing(query, limit, selected_store, selected_vectors):
                if selected_store is server.documents_store:
                    raise OSError("documents unavailable")
                return []

            server.hybrid_search = failing
            status, body = self.request("GET", "/unified/search?q=x&scope=all")
            self.assertEqual(status, 200)
            self.assertIn("documents: search backend unavailable", body["warnings"])
        finally:
            server.hybrid_search = original

    def test_documents_reconcile_uses_controlled_contract(self):
        self.assertEqual(self.request("POST", "/documents/reconcile", {})[0], 400)

    def test_document_status_vector_diagnostics_and_error(self):
        original = server.documents_vector_store
        server.documents_vector_store = FakeVectors()
        try:
            status, body = self.request("GET", "/documents/status")
            self.assertEqual(status, 200)
            self.assertIn("vectors", body)
            server.documents_vector_store.diagnostics = lambda store: (_ for _ in ()).throw(
                OSError("down")
            )
            status, body = self.request("GET", "/documents/status")
            self.assertEqual(body["vectors"]["status"], "error")
        finally:
            server.documents_vector_store = original


class DocumentConfigurationTests(unittest.TestCase):
    def test_document_scope_rejects_missing_configuration(self):
        with patch.object(server, "documents_store", None):
            with self.assertRaises(LookupError):
                server.unified_search("x", 2, "documents")

    def test_invalid_document_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            server.unified_search("x", 2, "document")

    def test_prompt_profile_applies_document_quota(self):
        rows = [
            {"source": "document", "relevance_score": 0.9, "id": "d1"},
            {"source": "document", "relevance_score": 0.8, "id": "d2"},
            {"source": "document", "relevance_score": 0.7, "id": "d3"},
        ]
        with patch.object(server, "PROMPT_SOURCE_QUOTAS", {"document": 2}):
            selected = server.apply_search_profile(rows, 10, "prompt")
        self.assertEqual([row["id"] for row in selected], ["d1", "d2"])

    def test_unconfigured_document_routes_return_404(self):
        previous = (server.DOCUMENTS_ROOT, server.documents_store)
        server.DOCUMENTS_ROOT = None
        server.documents_store = None
        http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        Thread(target=http.serve_forever, daemon=True).start()
        try:
            for method, path in (
                ("GET", "/documents/status"),
                ("GET", "/documents/search?q=x"),
                ("POST", "/documents/index"),
            ):
                connection = HTTPConnection(*http.server_address)
                connection.request(method, path)
                self.assertEqual(connection.getresponse().status, 404)
        finally:
            http.shutdown()
            server.DOCUMENTS_ROOT, server.documents_store = previous
