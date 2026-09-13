import json
import sys
import tempfile
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
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

    def test_stable_read_raises_when_file_keeps_changing(self):
        path = self.root / "flaky.md"
        path.write_text("unstable content")
        calls = {"n": 0}
        real_stat = Path.stat
        real_read = Path.read_text

        def flaky_stat(self, *args, **kwargs):
            # Alternate the reported size so the before/after comparison
            # never stabilises across the 3 attempts.
            calls["n"] += 1
            st = real_stat(self, *args, **kwargs)
            return SimpleNamespace(
                st_size=st.st_size + calls["n"],
                st_mtime_ns=st.st_mtime_ns,
                st_mode=st.st_mode,
            )

        def flaky_read(self, *args, **kwargs):
            return real_read(self, *args, **kwargs)

        with patch.object(Path, "stat", flaky_stat), patch.object(Path, "read_text", flaky_read):
            result = self.indexer.scan()
        self.assertEqual(result.errors, 1)
        self.assertEqual(self.indexer.status()["files_with_errors"], 1)

    def test_sqlite_failure_without_vector_store_skips_cleanup(self):
        indexer = DocumentIndexer(
            self.root,
            self.store,
            vector_store=None,
            embed=lambda text: [0.125] * 8,
        )
        path = self.root / "nov.md"
        path.write_text("no vector backend")
        indexer.scan()
        path.write_text("changed body")
        original = self.store.upsert_file_precomputed

        def fail(*args):
            raise OSError("sqlite full")

        self.store.upsert_file_precomputed = fail
        try:
            result = indexer.scan()
        finally:
            self.store.upsert_file_precomputed = original
        self.assertEqual(result.errors, 1)

    def test_errors_returns_structured_diagnostics_for_failed_files(self):
        (self.root / "bad.md").write_bytes(b"\xff")
        (self.root / "good.md").write_text("good content")
        self.indexer.scan()
        errors = self.indexer.errors()
        self.assertEqual(len(errors), 1)
        error = errors[0]
        self.assertEqual(error["path"], "bad.md")
        self.assertEqual(error["status"], "error")
        self.assertIn("error_class", error)
        self.assertIn("error_message", error)
        self.assertIsNotNone(error["last_attempt_at"])

    def test_errors_exposes_indexing_failures_with_error_class(self):
        path = self.root / "fail.md"
        path.write_text("indexable")
        self.indexer.scan()
        path.write_text("new content that will fail")
        self.vectors.fail = True
        self.indexer.scan()
        errors = self.indexer.errors()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["path"], "fail.md")
        self.assertEqual(errors[0]["status"], "error")
        self.assertTrue(len(errors[0]["error_message"]) > 0)

    def test_errors_strips_absolute_paths_from_messages(self):
        sanitized = DocumentIndexer._sanitize_error_row(
            (
                "rel/file.md",
                "OSError: /secret/data/path/file.txt not found",
                "2026-01-01T00:00:00",
                None,
                "error",
            )
        )
        self.assertNotIn("/secret", sanitized["error_message"])
        self.assertEqual(sanitized["error_class"], "OSError")
        self.assertIn("<path>", sanitized["error_message"])

    def test_errors_truncates_long_messages(self):
        long_msg = "x" * 5000
        sanitized = DocumentIndexer._sanitize_error_row(
            ("file.md", long_msg, "2026-01-01T00:00:00", None, "error")
        )
        self.assertLessEqual(len(sanitized["error_message"]), 1000)

    def test_errors_limit_validation(self):
        with self.assertRaises(ValueError):
            self.indexer.errors(limit=0)
        with self.assertRaises(ValueError):
            self.indexer.errors(limit=1001)

    def test_status_includes_error_details(self):
        (self.root / "broken.md").write_bytes(b"\xff")
        self.indexer.scan()
        status = self.indexer.status()
        self.assertEqual(status["files_with_errors"], 1)
        self.assertEqual(len(status["errors"]), 1)
        self.assertEqual(status["errors"][0]["path"], "broken.md")

    def test_stale_error_state_is_retried_when_content_is_unchanged(self):
        """A transient failure recorded after a successful index must not pin a
        file at ``status='error'`` forever (regression: U8 document error)."""
        path = self.root / "u8.md"
        path.write_text("# U8\nunchanged searchable body")
        self.indexer.scan()
        self.assertEqual(self.indexer.status()["files_with_errors"], 0)

        # Reproduce the live symptom: the stored digest stays valid while a later
        # attempt fails, leaving ``last_success_at`` set but ``status='error'``.
        self.indexer._state(
            "u8.md",
            digest=self.store.file_digest("u8.md"),
            status="error",
            error="OperationalError: cannot commit - no transaction is active",
        )
        self.assertEqual(self.indexer.status()["files_with_errors"], 1)

        result = self.indexer.scan()
        self.assertEqual(result.errors, 0)
        self.assertEqual(result.unchanged, 0)
        self.assertEqual(result.changed, 1)
        self.assertEqual(self.indexer.status()["files_with_errors"], 0)
        self.assertEqual(self.store.search("unchanged", 1)[0]["path"], "u8.md")

    def test_persistent_failure_stays_recorded_across_scans(self):
        """Retrying errored paths must not hide a file that never indexes."""
        (self.root / "broken.md").write_bytes(b"\xff")
        self.indexer.scan()
        self.assertEqual(self.indexer.status()["files_with_errors"], 1)
        second = self.indexer.scan()
        self.assertEqual(second.errors, 1)
        self.assertEqual(self.indexer.status()["files_with_errors"], 1)


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

    def test_document_errors_endpoint_returns_empty_when_no_failures(self):
        self.request("POST", "/documents/index", {"max_files": 10})
        status, body = self.request("GET", "/documents/errors")
        self.assertEqual(status, 200)
        self.assertIsInstance(body["errors"], list)

    def test_document_errors_endpoint_exposes_failed_files(self):
        (self.root / "api-bad.md").write_bytes(b"\xff")
        self.request("POST", "/documents/index", {"max_files": 10})
        status, body = self.request("GET", "/documents/errors")
        self.assertEqual(status, 200)
        paths = [item["path"] for item in body["errors"]]
        self.assertIn("api-bad.md", paths)
        for item in body["errors"]:
            self.assertIn("status", item)
            self.assertIn("error_class", item)
            self.assertIn("error_message", item)
            self.assertIn("last_attempt_at", item)
            self.assertIn("last_success_at", item)

    def test_document_errors_limit_validation(self):
        status, body = self.request("GET", "/documents/errors?limit=0")
        self.assertEqual(status, 400)
        status, body = self.request("GET", "/documents/errors?limit=abc")
        self.assertEqual(status, 400)

    def test_document_status_includes_error_list(self):
        (self.root / "status-err.md").write_bytes(b"\xff")
        self.request("POST", "/documents/index", {"max_files": 10})
        status, body = self.request("GET", "/documents/status")
        self.assertEqual(status, 200)
        self.assertIn("errors", body)
        self.assertIsInstance(body["errors"], list)
        self.assertGreaterEqual(body["files_with_errors"], 1)


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
                ("GET", "/documents/errors"),
                ("GET", "/documents/search?q=x"),
                ("POST", "/documents/index"),
            ):
                connection = HTTPConnection(*http.server_address)
                connection.request(method, path)
                self.assertEqual(connection.getresponse().status, 404)
        finally:
            http.shutdown()
            server.DOCUMENTS_ROOT, server.documents_store = previous
