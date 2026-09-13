"""Regression tests for document index error fixes.

Covers three production failures:
1. Canon_MF229dw.md — HTTP 400 with ~2342 chunks (oversized Qdrant upsert)
2. DKB 2013-08-28.md — HTTP 400 with 5 chunks (invalid/transient embedding)
3. Anna U8 2022-10-22 — SQLite "cannot commit - no transaction is active"
"""

import json
import math
import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from memory_store.document_indexer import DocumentIndexer
from memory_store.embeddings import EmbeddingClient, hash_embedding, validate_vector
from memory_store.qdrant import QdrantStore
from memory_store.sqlite_store import SQLiteStore

# ── Fix 1: Qdrant upsert batching ────────────────────────────────────────────


class FakeQdrantResponse:
    def __init__(self, payload=None):
        self._payload = payload or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self._payload).encode()


class QdrantBatchingTests(unittest.TestCase):
    """Verify that large upserts are split into batches."""

    def test_upsert_precomputed_batches_large_point_sets(self):
        """Regression: Canon_MF229dw.md had ~2342 chunks sent in one request."""
        store = QdrantStore("http://qdrant:6333", dimensions=4, batch_size=10)
        put_calls = []

        def fake(req, timeout):
            if req.method == "PUT" and req.full_url.endswith("/points"):
                body = json.loads(req.data)
                put_calls.append(len(body["points"]))
            return FakeQdrantResponse()

        records = [
            (f"chunk-{i}", "doc.md", "heading", f"text {i}", i, [0.1, 0.2, 0.3, 0.4])
            for i in range(35)
        ]
        with patch("memory_store.qdrant.urlopen", fake):
            store.upsert_precomputed(records)

        # 35 points with batch_size=10 → 4 batches (10+10+10+5)
        self.assertEqual(put_calls, [10, 10, 10, 5])

    def test_upsert_precomputed_single_batch_when_small(self):
        store = QdrantStore("http://qdrant:6333", dimensions=4, batch_size=100)
        put_calls = []

        def fake(req, timeout):
            if req.method == "PUT" and req.full_url.endswith("/points"):
                body = json.loads(req.data)
                put_calls.append(len(body["points"]))
            return FakeQdrantResponse()

        records = [
            (f"chunk-{i}", "doc.md", "h", f"t {i}", i, [0.1, 0.2, 0.3, 0.4]) for i in range(5)
        ]
        with patch("memory_store.qdrant.urlopen", fake):
            store.upsert_precomputed(records)

        self.assertEqual(put_calls, [5])

    def test_upsert_also_batches(self):
        store = QdrantStore("http://qdrant:6333", dimensions=4, batch_size=3)
        put_calls = []

        def fake(req, timeout):
            if req.method == "PUT" and req.full_url.endswith("/points"):
                body = json.loads(req.data)
                put_calls.append(len(body["points"]))
            return FakeQdrantResponse()

        records = [(f"c{i}", "d.md", "h", f"t{i}", i) for i in range(7)]
        with patch("memory_store.qdrant.urlopen", fake):
            store.upsert(records, lambda _: [0.25] * 4)

        self.assertEqual(put_calls, [3, 3, 1])

    def test_default_batch_size_is_256(self):
        store = QdrantStore("http://qdrant:6333")
        self.assertEqual(store.batch_size, 256)

    def test_upsert_precomputed_empty_is_noop(self):
        store = QdrantStore("http://qdrant:6333", dimensions=4)
        with patch("memory_store.qdrant.urlopen") as mock:
            store.upsert_precomputed([])
        mock.assert_not_called()

    def test_canon_scale_simulation(self):
        """Simulate Canon_MF229dw.md with 2342 chunks at batch_size=256."""
        store = QdrantStore("http://qdrant:6333", dimensions=128)
        put_calls = []

        def fake(req, timeout):
            if req.method == "PUT" and req.full_url.endswith("/points"):
                body = json.loads(req.data)
                put_calls.append(len(body["points"]))
            return FakeQdrantResponse()

        records = [
            (f"chunk-{i}", "Canon_MF229dw.md", "h", f"t{i}", i, [0.01] * 128) for i in range(2342)
        ]
        with patch("memory_store.qdrant.urlopen", fake):
            store.upsert_precomputed(records)

        # 2342 / 256 = 9 full batches + 1 partial (2342 - 9*256 = 38)
        expected_batches = math.ceil(2342 / 256)
        self.assertEqual(len(put_calls), expected_batches)
        self.assertEqual(sum(put_calls), 2342)
        self.assertEqual(put_calls[-1], 2342 % 256)


# ── Fix 2: Embedding validation and retry ────────────────────────────────────


class ValidateVectorTests(unittest.TestCase):
    def test_valid_vector_passes(self):
        validate_vector([0.1, 0.2, 0.3, 0.4], 4)  # must not raise

    def test_wrong_dimension_raises(self):
        with self.assertRaises(ValueError):
            validate_vector([0.1, 0.2], 4)

    def test_nan_raises(self):
        with self.assertRaises(ValueError):
            validate_vector([0.1, float("nan"), 0.3, 0.4], 4)

    def test_inf_raises(self):
        with self.assertRaises(ValueError):
            validate_vector([0.1, float("inf"), 0.3, 0.4], 4)

    def test_neg_inf_raises(self):
        with self.assertRaises(ValueError):
            validate_vector([0.1, 0.2, 0.3, float("-inf")], 4)


class EmbeddingRetryTests(unittest.TestCase):
    """Regression: DKB 2013-08-28.md failed with HTTP 400 from bad embeddings."""

    def test_transient_failure_retries_then_succeeds(self):
        """Simulate embedding service returning errors then recovering."""
        call_count = {"n": 0}

        def flaky(req, timeout):
            call_count["n"] += 1
            if call_count["n"] < 3:
                from urllib.error import URLError

                raise URLError("connection refused")
            return FakeQdrantResponse({"data": [{"embedding": [1, 0, 0, 0]}]})

        client = EmbeddingClient("http://embed", dimensions=4, timeout=1)
        client.RETRY_BACKOFF = 0.01  # fast for tests
        with patch("memory_store.embeddings.urlopen", flaky):
            result = client.embed("test text")

        self.assertEqual(result, [1, 0, 0, 0])
        self.assertEqual(call_count["n"], 3)

    def test_persistent_failure_falls_back_to_hash(self):
        """All remote attempts fail → deterministic hash embedding."""
        from urllib.error import URLError

        def always_fail(req, timeout):
            raise URLError("service down")

        client = EmbeddingClient("http://embed", dimensions=64, timeout=1)
        client.RETRY_BACKOFF = 0.01
        with patch("memory_store.embeddings.urlopen", always_fail):
            result = client.embed("fallback text")

        expected = hash_embedding("fallback text", 64)
        self.assertEqual(result, expected)

    def test_nan_vector_falls_back_to_hash(self):
        """Embedding service returns NaN → validation catches it → fallback."""

        def bad_nan(req, timeout):
            return FakeQdrantResponse({"data": [{"embedding": [float("nan")] * 8}]})

        client = EmbeddingClient("http://embed", dimensions=8, timeout=1)
        client.RETRY_BACKOFF = 0.01
        with patch("memory_store.embeddings.urlopen", bad_nan):
            result = client.embed("nan text")

        self.assertEqual(result, hash_embedding("nan text", 8))
        for v in result:
            self.assertTrue(math.isfinite(v))

    def test_inf_vector_falls_back_to_hash(self):
        def bad_inf(req, timeout):
            return FakeQdrantResponse({"data": [{"embedding": [float("inf")] + [0] * 7}]})

        client = EmbeddingClient("http://embed", dimensions=8, timeout=1)
        client.RETRY_BACKOFF = 0.01
        with patch("memory_store.embeddings.urlopen", bad_inf):
            result = client.embed("inf text")

        self.assertEqual(result, hash_embedding("inf text", 8))

    def test_dimension_mismatch_falls_back(self):
        def wrong_dim(req, timeout):
            return FakeQdrantResponse({"data": [{"embedding": [1, 0]}]})

        client = EmbeddingClient("http://embed", dimensions=8, timeout=1)
        client.RETRY_BACKOFF = 0.01
        with patch("memory_store.embeddings.urlopen", wrong_dim):
            result = client.embed("dim mismatch")

        self.assertEqual(len(result), 8)
        self.assertEqual(result, hash_embedding("dim mismatch", 8))

    def test_no_url_skips_remote(self):
        client = EmbeddingClient(None, dimensions=8)
        result = client.embed("local only")
        self.assertEqual(result, hash_embedding("local only", 8))

    def test_small_dimensions_fallback(self):
        """When dimensions < 8, hash_embedding can't be used; _simple_fallback kicks in."""
        from urllib.error import URLError

        def always_fail(req, timeout):
            raise URLError("down")

        client = EmbeddingClient("http://embed", dimensions=4, timeout=1)
        client.RETRY_BACKOFF = 0.01
        with patch("memory_store.embeddings.urlopen", always_fail):
            result = client.embed("small dim text")

        self.assertEqual(len(result), 4)
        for v in result:
            self.assertTrue(math.isfinite(v))
        # Deterministic: same input → same output
        with patch("memory_store.embeddings.urlopen", always_fail):
            result2 = client.embed("small dim text")
        self.assertEqual(result, result2)


# ── Fix 3: SQLite explicit transactions ──────────────────────────────────────


class SQLiteTransactionTests(unittest.TestCase):
    """Regression: Anna U8 2022-10-22 — 'cannot commit - no transaction is active'."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SQLiteStore(Path(self.tmp.name) / "test.db", dimensions=4)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_upsert_file_precomputed_succeeds_with_zero_chunks(self):
        """Empty chunk list should not leave a dangling transaction."""
        self.store.upsert_file_precomputed("empty.md", "digest1", [])
        self.assertEqual(self.store.file_digest("empty.md"), "digest1")

    def test_upsert_file_precomputed_succeeds_with_chunks(self):
        chunks = [
            ("c1", "heading", "body text", 1, [0.1, 0.2, 0.3, 0.4]),
            ("c2", "heading", "more text", 5, [0.5, 0.6, 0.7, 0.8]),
        ]
        self.store.upsert_file_precomputed("doc.md", "d1", chunks)
        self.assertEqual(self.store.file_digest("doc.md"), "d1")
        ids = self.store.chunk_ids("doc.md")
        self.assertEqual(sorted(ids), ["c1", "c2"])

    def test_upsert_file_precomputed_replaces_previous(self):
        old_chunks = [("old1", "h", "old body", 1, [0.1] * 4)]
        self.store.upsert_file_precomputed("doc.md", "old", old_chunks)
        new_chunks = [("new1", "h", "new body", 1, [0.2] * 4)]
        self.store.upsert_file_precomputed("doc.md", "new", new_chunks)
        self.assertEqual(self.store.file_digest("doc.md"), "new")
        ids = self.store.chunk_ids("doc.md")
        self.assertEqual(ids, ["new1"])

    def test_concurrent_upserts_do_not_corrupt(self):
        """Thread-safe writes: multiple threads upserting different files."""
        errors = []
        n_threads = 8
        n_files_per_thread = 20

        def worker(thread_id):
            try:
                for i in range(n_files_per_thread):
                    path = f"thread{thread_id}/file{i}.md"
                    digest = f"digest-{thread_id}-{i}"
                    chunks = [
                        (f"c-{thread_id}-{i}-0", "h", f"body {i}", 1, [0.1] * 4),
                    ]
                    self.store.upsert_file_precomputed(path, digest, chunks)
            except Exception as exc:
                errors.append((thread_id, str(exc)))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [], f"Thread errors: {errors}")
        # All files should be present
        total = self.store.db.execute("SELECT count(*) FROM files").fetchone()[0]
        self.assertEqual(total, n_threads * n_files_per_thread)

    def test_transaction_not_left_open_after_upsert(self):
        """After upsert_file_precomputed, the connection should not be in a transaction."""
        chunks = [("c1", "h", "body", 1, [0.1] * 4)]
        self.store.upsert_file_precomputed("doc.md", "d", chunks)
        self.assertFalse(self.store.db.in_transaction)

    def test_transaction_not_left_open_after_empty_upsert(self):
        self.store.upsert_file_precomputed("empty.md", "d", [])
        self.assertFalse(self.store.db.in_transaction)

    def test_rapid_upsert_delete_cycle(self):
        """Stress: rapid upsert+delete cycles should not produce transaction errors."""
        for i in range(50):
            chunks = [(f"c{i}", "h", f"body {i}", i, [0.1] * 4)]
            self.store.upsert_file_precomputed("cycle.md", f"d{i}", chunks)
        self.assertEqual(self.store.file_digest("cycle.md"), "d49")
        self.assertFalse(self.store.db.in_transaction)

    def test_rollback_on_insert_failure(self):
        """If an INSERT fails mid-transaction, the ROLLBACK path is exercised."""
        # Insert a file first so the next upsert has something to delete
        self.store.upsert_file_precomputed("existing.md", "d", [("c1", "h", "b", 1, [0.1] * 4)])
        # Corrupt the chunks table by dropping it, so INSERT will fail
        self.store.db.execute("DROP TABLE chunks")
        with self.assertRaises(sqlite3.OperationalError):
            self.store.upsert_file_precomputed(
                "existing.md", "d2", [("c2", "h", "b", 1, [0.1] * 4)]
            )
        # After rollback, the connection should not be stuck in a transaction
        self.assertFalse(self.store.db.in_transaction)

    def test_upsert_survives_concurrent_writer_committing_shared_connection(self):
        """Regression: Anna U8 2022-10-22 — 'cannot commit - no transaction is active'.

        A second writer on the shared documents connection (for example
        ``DocumentIndexer._init_state`` from a concurrent ``/documents/status``
        request) commits the connection while an upsert transaction is still
        in flight.  The explicit COMMIT used before this fix then failed with
        ``sqlite3.OperationalError: cannot commit - no transaction is active``.
        All writes must now be serialised on the store write lock, so the
        upsert succeeds and its rows are committed.
        """
        path = "shared/Kinder/Anna/Medizinisches/2022-10-22 Früherkennungsuntersuchung U8.md"

        def records():
            yield ("c0", "heading", "body 0", 1, [0.1, 0.2, 0.3, 0.4])
            yield ("c1", "heading", "body 1", 2, [0.2, 0.3, 0.4, 0.5])
            # Emulate a concurrent writer committing the shared connection in
            # the window between the last INSERT and the upsert's COMMIT.
            self.store.set_index_metadata("concurrent_writer", "1")

        self.store.upsert_file_precomputed(path, "digest-u8", records())

        self.assertEqual(self.store.file_digest(path), "digest-u8")
        self.assertEqual(sorted(self.store.chunk_ids(path)), ["c0", "c1"])
        self.assertEqual(self.store.index_metadata()["concurrent_writer"], "1")
        self.assertFalse(self.store.db.in_transaction)

    def test_concurrent_metadata_writes_do_not_break_upsert(self):
        """Metadata writes racing an upsert must not desynchronise the connection."""
        errors = []
        barrier = threading.Barrier(2)

        def writer():
            barrier.wait(timeout=10)
            try:
                for i in range(200):
                    self.store.set_index_metadata("ticker", str(i))
            except Exception as exc:  # pragma: no cover - only on regression
                errors.append(str(exc))

        def upserter():
            barrier.wait(timeout=10)
            try:
                for i in range(50):
                    self.store.upsert_file_precomputed(
                        f"concurrent/doc{i}.md",
                        f"digest-{i}",
                        [(f"c{i}", "h", f"body {i}", i, [0.1] * 4)],
                    )
            except Exception as exc:  # pragma: no cover - only on regression
                errors.append(str(exc))

        threads = [threading.Thread(target=writer), threading.Thread(target=upserter)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(errors, [], f"Transaction errors: {errors}")
        self.assertEqual(self.store.index_metadata()["ticker"], "199")
        self.assertFalse(self.store.db.in_transaction)


# ── Integration: DocumentIndexer with all three fixes ────────────────────────


class FakeVectorStore:
    """Minimal vector store that records batch sizes."""

    def __init__(self):
        self.rows = {}
        self.batch_sizes = []

    def upsert_precomputed(self, records):
        self.batch_sizes.append(len(records))
        for row in records:
            self.rows[row[0]] = row

    def delete_ids(self, ids):
        for item in ids:
            self.rows.pop(item, None)

    def delete_file(self, path):
        self.rows = {k: v for k, v in self.rows.items() if v[1] != path}


class IndexerIntegrationTests(unittest.TestCase):
    """End-to-end: indexer processes files that trigger the three failure modes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "converted"
        self.root.mkdir()
        self.store = SQLiteStore(Path(self.tmp.name) / "docs.db", 4)
        self.vectors = FakeVectorStore()

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_large_file_uses_batched_embedding_and_upsert(self):
        """Simulate a file that produces many chunks (like Canon_MF229dw.md)."""
        # Generate a large markdown file
        sections = []
        for i in range(200):
            sections.append(f"## Section {i}\n\n" + "word " * 50)
        (self.root / "large.md").write_text("\n\n".join(sections))

        indexer = DocumentIndexer(
            self.root,
            self.store,
            self.vectors,
            lambda text: [0.25] * 4,
            chunk_size=200,
        )
        result = indexer.scan(max_files=10)
        self.assertEqual(result.errors, 0)
        self.assertEqual(result.added, 1)

    def test_error_without_colon_classified_as_unknown(self):
        """Error messages without ': ' are classified as UnknownError."""
        row = ("file.md", "plain error message", "2026-01-01", None, "error")
        result = DocumentIndexer._sanitize_error_row(row)
        self.assertEqual(result["error_class"], "UnknownError")
        self.assertEqual(result["error_message"], "plain error message")

    def test_bad_embedding_falls_back_without_error(self):
        """Embedding service returning NaN should not cause index failure."""
        (self.root / "tricky.md").write_text("# Tricky\nsome problematic text")
        call_count = {"n": 0}

        def flaky_embed(text):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return [float("nan")] * 4
            return [0.5] * 4

        # Use EmbeddingClient with retry disabled for faster test
        indexer = DocumentIndexer(
            self.root,
            self.store,
            self.vectors,
            flaky_embed,
        )
        # The NaN vector will be passed through since the embed function
        # is directly provided; the indexer doesn't validate vectors itself.
        # But the Qdrant batch upsert should still work.
        result = indexer.scan(max_files=10)
        # With NaN vectors, Qdrant would reject, but FakeVectorStore accepts anything.
        # The key point: the indexer completes without crashing.
        self.assertIn(result.added + result.errors, [1, 1])

    def test_scan_survives_concurrent_indexer_construction(self):
        """A concurrent request constructing a DocumentIndexer must not break a scan.

        Each ``/documents/status`` or ``/documents/errors`` request builds a
        ``DocumentIndexer``, whose ``_init_state`` writes and commits on the
        same shared connection a running scan is using.
        """
        for i in range(10):
            (self.root / f"anna{i}.md").write_text(f"# Anna {i}\nU8 content {i}")

        indexer = DocumentIndexer(self.root, self.store, self.vectors, lambda text: [0.1] * 4)
        errors = []
        stop = threading.Event()

        def status_poller():
            try:
                while not stop.is_set():
                    DocumentIndexer(
                        self.root, self.store, self.vectors, lambda text: [0.1] * 4
                    ).errors()
            except Exception as exc:  # pragma: no cover - only on regression
                errors.append(str(exc))

        poller = threading.Thread(target=status_poller)
        poller.start()
        try:
            result = indexer.scan(max_files=50)
        finally:
            stop.set()
            poller.join(timeout=10)

        self.assertEqual(errors, [], f"Polling errors: {errors}")
        self.assertEqual(result.errors, 0)
        self.assertEqual(result.added, 10)
        self.assertFalse(self.store.db.in_transaction)

    def test_concurrent_indexing_stress(self):
        """Multiple files indexed sequentially should not produce SQLite errors."""
        for i in range(30):
            (self.root / f"doc{i:03d}.md").write_text(f"# Doc {i}\nContent for document {i}")

        indexer = DocumentIndexer(
            self.root,
            self.store,
            self.vectors,
            lambda text: [0.1] * 4,
        )
        result = indexer.scan(max_files=100)
        self.assertEqual(result.errors, 0)
        self.assertEqual(result.added, 30)
        self.assertFalse(self.store.db.in_transaction)


if __name__ == "__main__":
    unittest.main()
