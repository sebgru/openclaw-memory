"""Tests for index freshness metadata, vector diagnostics, and reconciliation."""

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
from memory_store.indexer import Indexer
from memory_store.qdrant import QdrantStore
from memory_store.sqlite_store import SQLiteStore


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self._payload).encode()


class IndexMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = SQLiteStore()

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_status_reports_index_metadata(self):
        (self.root / "a.md").write_text("# A\ncontent")
        Indexer(self.root, self.store).scan()
        index = self.store.status()["index"]
        self.assertIn("last_index_started_at", index)
        self.assertIn("last_index_completed_at", index)
        self.assertEqual(index["last_index_error"], "")
        self.assertIsNotNone(index["age_seconds"])
        self.assertGreaterEqual(index["age_seconds"], 0.0)

    def test_missing_root_records_error(self):
        with self.assertRaises(FileNotFoundError):
            Indexer(self.root / "missing", self.store).scan()
        self.assertIn("does not exist", self.store.status()["index"]["last_index_error"])

    def test_scan_failure_records_error(self):
        (self.root / "a.md").write_text("# A\ncontent")
        with patch.object(Indexer, "_scan", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                Indexer(self.root, self.store).scan()
        self.assertIn("boom", self.store.status()["index"]["last_index_error"])

    def test_status_handles_malformed_timestamp(self):
        self.store.set_index_metadata("last_index_completed_at", "not-a-date")
        self.assertIsNone(self.store.status()["index"]["age_seconds"])


class QdrantDiagnosticsTests(unittest.TestCase):
    def make_store(self):
        return QdrantStore("http://qdrant:6333", collection="mem", dimensions=2)

    def setUp(self):
        self.q = self.make_store()
        self.sql = SQLiteStore()
        self.sql.upsert_file("a.md", "d", [("id1", "H", "body", 1)])

    def tearDown(self):
        self.sql.close()

    def test_point_count(self):
        with patch(
            "memory_store.qdrant.urlopen",
            lambda req, timeout: FakeResponse({"result": {"count": 7}}),
        ):
            self.assertEqual(self.q.point_count(), 7)

    def test_point_ids_empty(self):
        self.assertEqual(self.q.point_ids([]), set())

    def test_point_ids_returns_present(self):
        with patch(
            "memory_store.qdrant.urlopen",
            lambda req, timeout: FakeResponse({"result": [{"id": "abc"}]}),
        ):
            self.assertEqual(self.q.point_ids(["id1"]), {"abc"})

    def test_diagnostics_reports_missing_vectors(self):
        def fake(req, timeout):
            if req.full_url.endswith("/points/count"):
                return FakeResponse({"result": {"count": 0}})
            return FakeResponse({"result": []})

        with patch("memory_store.qdrant.urlopen", fake):
            result = self.q.diagnostics(self.sql)
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["points"], 0)
        self.assertEqual(result["missing_vectors"], 1)

    def test_reconcile_upserts_missing_points(self):
        calls = []

        def fake(req, timeout):
            calls.append((req.method, req.full_url, json.loads(req.data) if req.data else None))
            if req.full_url.endswith("/points") and req.method == "POST":
                return FakeResponse({"result": []})
            return FakeResponse({})

        with patch("memory_store.qdrant.urlopen", fake):
            result = self.q.reconcile_missing(self.sql, lambda _: [0.0, 0.0], limit=10)
        self.assertEqual(result, {"reconciled": 1, "remaining": 0})
        upserts = [c for c in calls if c[0] == "PUT"]
        self.assertEqual(len(upserts), 1)

    def test_reconcile_respects_limit(self):
        self.sql.upsert_file("b.md", "d2", [("id2", "H", "body", 1)])

        def fake(req, timeout):
            if req.full_url.endswith("/points") and req.method == "POST":
                return FakeResponse({"result": []})
            return FakeResponse({})

        with patch("memory_store.qdrant.urlopen", fake):
            result = self.q.reconcile_missing(self.sql, lambda _: [0.0, 0.0], limit=1)
        self.assertEqual(result, {"reconciled": 1, "remaining": 1})

    def test_reconcile_no_missing_is_noop(self):
        import uuid

        expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "id1"))

        def fake(req, timeout):
            if req.full_url.endswith("/points") and req.method == "POST":
                return FakeResponse({"result": [{"id": expected_id}]})
            return FakeResponse({})

        with patch("memory_store.qdrant.urlopen", fake):
            result = self.q.reconcile_missing(self.sql, lambda _: [0.0, 0.0])
        self.assertEqual(result, {"reconciled": 0, "remaining": 0})


class ReconcileEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.prev_root = server.ROOT
        server.ROOT = str(Path(cls.tmp.name))
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        Thread(target=cls.http.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        server.ROOT = cls.prev_root
        cls.tmp.cleanup()

    def request(self, method, path, body=None):
        c = HTTPConnection(*self.http.server_address)
        c.request(method, path, body=body)
        r = c.getresponse()
        data = r.read()
        return r.status, json.loads(data) if data else None

    def test_reconcile_requires_confirm(self):
        status, body = self.request("POST", "/reconcile", json.dumps({}))
        self.assertEqual(status, 400)
        self.assertIn("confirm", body["error"])

    def test_reconcile_without_vector_backend_returns_404(self):
        status, body = self.request("POST", "/reconcile", json.dumps({"confirm": True}))
        self.assertEqual(status, 404)
        self.assertIn("not configured", body["error"])

    def test_reconcile_invalid_limit_returns_400(self):
        original = server.vector_store
        server.vector_store = object()
        try:
            status, body = self.request(
                "POST", "/reconcile", json.dumps({"confirm": True, "limit": 0})
            )
        finally:
            server.vector_store = original
        self.assertEqual(status, 400)
        self.assertIn("limit", body["error"])

    def test_reconcile_invalid_limit_upper_bound_returns_400(self):
        original = server.vector_store
        server.vector_store = object()
        try:
            status, body = self.request(
                "POST", "/reconcile", json.dumps({"confirm": True, "limit": 10001})
            )
        finally:
            server.vector_store = original
        self.assertEqual(status, 400)
        self.assertIn("limit", body["error"])

    def test_reconcile_success(self):
        class FakeVectors:
            def reconcile_missing(self, store, embed, limit):
                return {"reconciled": 2, "remaining": 0}

        original = server.vector_store
        server.vector_store = FakeVectors()
        try:
            status, body = self.request("POST", "/reconcile", json.dumps({"confirm": True}))
        finally:
            server.vector_store = original
        self.assertEqual(status, 200)
        self.assertEqual(body, {"reconciled": 2, "remaining": 0})

    def test_archive_reconcile_without_archive_returns_404(self):
        status, body = self.request("POST", "/archive/reconcile", json.dumps({"confirm": True}))
        self.assertEqual(status, 404)

    def test_archive_status_includes_vector_diagnostics(self):
        class FakeVectors:
            def diagnostics(self, store):
                return {"points": 1, "chunks": 1, "missing_vectors": 0}

        archive_db = SQLiteStore()
        previous = (server.ARCHIVE_ROOT, server.archive_store, server.archive_vector_store)
        server.ARCHIVE_ROOT = str(Path(self.tmp.name) / "archive")
        Path(server.ARCHIVE_ROOT).mkdir(exist_ok=True)
        server.archive_store = archive_db
        server.archive_vector_store = FakeVectors()
        try:
            status, body = self.request("GET", "/archive/status")
            self.assertEqual(status, 200)
            self.assertEqual(body["vectors"]["missing_vectors"], 0)

            class BrokenVectors:
                def diagnostics(self, store):
                    raise RuntimeError("qdrant down")

            server.archive_vector_store = BrokenVectors()
            status, body = self.request("GET", "/archive/status")
            self.assertEqual(status, 200)
            self.assertEqual(body["vectors"]["status"], "error")
        finally:
            server.ARCHIVE_ROOT, server.archive_store, server.archive_vector_store = previous
            archive_db.close()

    def test_status_includes_vector_diagnostics(self):
        class FakeVectors:
            def diagnostics(self, store):
                return {"points": 1, "chunks": 1, "missing_vectors": 0}

        original = server.vector_store
        server.vector_store = FakeVectors()
        try:
            status, body = self.request("GET", "/status")
        finally:
            server.vector_store = original
        self.assertEqual(status, 200)
        self.assertEqual(body["vectors"]["missing_vectors"], 0)

    def test_status_handles_vector_diagnostics_error(self):
        class FakeVectors:
            def diagnostics(self, store):
                raise RuntimeError("qdrant down")

        original = server.vector_store
        server.vector_store = FakeVectors()
        try:
            status, body = self.request("GET", "/status")
        finally:
            server.vector_store = original
        self.assertEqual(status, 200)
        self.assertEqual(body["vectors"]["status"], "error")


if __name__ == "__main__":
    unittest.main()
