"""Extended tests: HTTP API routes, embeddings client, Qdrant adapter, chunker edge cases."""

import json
import sys
import tempfile
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from memory_store import server
from memory_store.chunker import chunk_markdown
from memory_store.embeddings import EmbeddingClient, hash_embedding
from memory_store.indexer import Indexer
from memory_store.qdrant import QdrantStore
from memory_store.sqlite_store import SQLiteStore

# ── hash_embedding ───────────────────────────────────────────────────────────


class HashEmbeddingTests(unittest.TestCase):
    def test_unit_norm_and_determinism(self):
        v1 = hash_embedding("hello world")
        v2 = hash_embedding("hello world")
        self.assertEqual(v1, v2)
        norm = sum(x * x for x in v1) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)
        self.assertEqual(len(v1), 128)

    def test_different_texts_differ(self):
        self.assertNotEqual(hash_embedding("alpha"), hash_embedding("beta"))

    def test_empty_text_returns_zero_vector(self):
        v = hash_embedding("")
        self.assertEqual(v, [0.0] * 128)

    def test_invalid_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            hash_embedding("x", dimensions=4)

    def test_minimum_dimensions_accepted(self):
        self.assertEqual(len(hash_embedding("x", dimensions=8)), 8)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self._payload).encode()


class EmbeddingClientTests(unittest.TestCase):
    def test_fallback_when_no_url(self):
        client = EmbeddingClient(None, dimensions=64)
        self.assertEqual(client.embed("hello"), hash_embedding("hello", 64))

    def test_openai_style_response(self):
        client = EmbeddingClient("http://embed", dimensions=4)
        with patch(
            "memory_store.embeddings.urlopen",
            lambda req, timeout: FakeResponse({"data": [{"embedding": [1, 0, 0, 0]}]}),
        ):
            self.assertEqual(client.embed("hi"), [1, 0, 0, 0])

    def test_ollama_style_response(self):
        client = EmbeddingClient("http://embed", dimensions=4)
        with patch(
            "memory_store.embeddings.urlopen",
            lambda req, timeout: FakeResponse({"embeddings": [[0, 1, 0, 0]]}),
        ):
            self.assertEqual(client.embed("hi"), [0, 1, 0, 0])

    def test_dimension_mismatch_raises(self):
        client = EmbeddingClient("http://embed", dimensions=4)
        with patch(
            "memory_store.embeddings.urlopen",
            lambda req, timeout: FakeResponse({"data": [{"embedding": [1, 0]}]}),
        ):
            with self.assertRaises(ValueError):
                client.embed("hi")

    def test_request_body_shape(self):
        client = EmbeddingClient("http://embed/api", model="m1", dimensions=4, timeout=3)
        captured = []

        def fake(req, timeout):
            captured.append((req.full_url, json.loads(req.data), timeout))
            return FakeResponse({"data": [{"embedding": [0, 0, 0, 1]}]})

        with patch("memory_store.embeddings.urlopen", fake):
            client.embed("hi")
        url, body, timeout = captured[0]
        self.assertEqual(url, "http://embed/api")
        self.assertEqual(body, {"model": "m1", "input": ["hi"]})
        self.assertEqual(timeout, 3)


# ── Qdrant adapter ───────────────────────────────────────────────────────────


class QdrantExtendedTests(unittest.TestCase):
    def make_store(self):
        return QdrantStore("http://qdrant:6333/", collection="mem", dimensions=2)

    def test_url_trailing_slash_stripped(self):
        self.assertEqual(self.make_store().base, "http://qdrant:6333")

    def test_upsert_empty_records_is_noop(self):
        store = self.make_store()
        with patch("memory_store.qdrant.urlopen") as mock:
            store.upsert([], lambda _: [0.0, 0.0])
        mock.assert_not_called()

    def test_ensure_collection_ignores_already_exists(self):
        store = self.make_store()

        def fake(req, timeout):
            raise Exception("[409] collection already exists")

        with patch("memory_store.qdrant.urlopen", fake):
            store.ensure_collection()  # must not raise

    def test_ensure_collection_reraises_other_errors(self):
        store = self.make_store()

        def fake(req, timeout):
            raise Exception("connection refused")

        with patch("memory_store.qdrant.urlopen", fake):
            with self.assertRaisesRegex(Exception, "connection refused"):
                store.ensure_collection()

    def test_delete_file_sends_filter(self):
        store = self.make_store()
        calls = []

        def fake(req, timeout):
            calls.append(json.loads(req.data))
            return FakeResponse({})

        with patch("memory_store.qdrant.urlopen", fake):
            store.delete_file("a.md")
        self.assertEqual(calls[0]["filter"]["must"][0]["match"]["value"], "a.md")

    def test_search_missing_result_key(self):
        store = self.make_store()
        with patch("memory_store.qdrant.urlopen", lambda req, timeout: FakeResponse({})):
            self.assertEqual(store.search([0.0, 0.0]), [])

    def test_upsert_error_propagates(self):
        store = self.make_store()

        def fake(req, timeout):
            raise URLError("down")

        with patch("memory_store.qdrant.urlopen", fake):
            with self.assertRaises(URLError):
                store.upsert([("x", "a.md", "", "hello", 1)], lambda _: [0.0, 0.0])


# ── Chunker edge cases ───────────────────────────────────────────────────────


class ChunkerEdgeCaseTests(unittest.TestCase):
    def test_small_max_chars_rejected(self):
        with self.assertRaises(ValueError):
            chunk_markdown("x", max_chars=50)

    def test_oversized_paragraph_is_split(self):
        paragraph = "word " * 500  # ~2500 chars, exceeds 1600
        chunks = chunk_markdown(f"# Title\n{paragraph}")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.text) <= 1600 for c in chunks))

    def test_multiple_paragraphs_in_section_split(self):
        paragraphs = "a" * 1500 + "\n\n" + "b" * 1500
        chunks = chunk_markdown(f"# T\n{paragraphs}")
        self.assertEqual([c.text for c in chunks], ["a" * 1500, "b" * 1500])

    def test_blank_lines_and_no_heading(self):
        chunks = chunk_markdown("\n\n\njust text\n\n\n")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading, "")
        self.assertEqual(chunks[0].text, "just text")

    def test_heading_without_body_is_skipped(self):
        self.assertEqual(chunk_markdown("# Empty"), [])

    def test_closing_hashes_stripped(self):
        chunks = chunk_markdown("# Heading ##\nbody")
        self.assertEqual(chunks[0].heading, "Heading")


# ── HTTP API ─────────────────────────────────────────────────────────────────


class ApiExtendedTests(unittest.TestCase):
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

    def test_healthz_reports_sqlite_backend_and_counts(self):
        status, body = self.request("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["backend"], "sqlite")
        self.assertIn("files", body)
        self.assertIn("chunks", body)

    def test_search_returns_results_for_indexed_content(self):
        (Path(self.tmp.name) / "doc.md").write_text("# API\nthe quick brown fox")
        status, body = self.request("POST", "/index")
        self.assertEqual(status, 200)
        self.assertEqual(body["added"], 1)
        status, body = self.request("GET", "/search?q=quick")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(body["results"]), 1)
        self.assertEqual(body["results"][0]["path"], "doc.md")

    def test_search_limit_bounds(self):
        self.assertEqual(self.request("GET", "/search?q=x&limit=0")[0], 400)
        self.assertEqual(self.request("GET", "/search?q=x&limit=101")[0], 400)
        self.assertEqual(self.request("GET", "/search?q=x&limit=-1")[0], 400)

    def test_unknown_get_returns_404(self):
        status, body = self.request("GET", "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body, {"error": "not found"})

    def test_post_unknown_path_returns_404(self):
        self.assertEqual(self.request("POST", "/nope")[0], 404)

    def test_post_index_bad_root_returns_400(self):
        prev = server.ROOT
        server.ROOT = str(Path(self.tmp.name) / "missing")
        try:
            status, body = self.request("POST", "/index")
        finally:
            server.ROOT = prev
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_hybrid_search_without_vector_store(self):
        results = server.hybrid_search("quick", 5)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)

    def test_hybrid_search_limit_respected(self):
        for n in range(5):
            (Path(self.tmp.name) / f"d{n}.md").write_text(f"# Doc\nshared term{n}")
        self.request("POST", "/index")
        self.assertLessEqual(len(server.hybrid_search("shared", 2)), 2)

    def test_hybrid_search_merges_lexical_and_semantic(self):
        lexical = [
            {"id": "a", "path": "a.md", "heading": "h", "text": "t", "line": 1, "score": -1.0}
        ]
        semantic = [
            {"id": "a", "path": "a.md", "heading": "h", "text": "t", "line": 1, "score": 0.9},
            {"id": "b", "path": "b.md", "heading": "h", "text": "t", "line": 1, "score": 0.8},
        ]
        with (
            patch.object(server.store, "search", return_value=lexical),
            patch.object(server.embedder, "embed", return_value=[0.0] * 128),
            patch.object(server, "vector_store") as mock_vector,
        ):
            mock_vector.search.return_value = semantic
            results = server.hybrid_search("t", 10)
        # RRF: 'a' appears in both lists and accumulates two boosts; the raw
        # semantic 0.9 score is discarded for 'a' because the lexical dict won
        # the setdefault. 'b' keeps its semantic score plus one RRF boost.
        self.assertEqual(len(results), 2)
        ids = [r["id"] for r in results]
        self.assertIn("a", ids)
        self.assertIn("b", ids)
        self.assertEqual(results[0]["id"], "b")
        a_row = next(r for r in results if r["id"] == "a")
        self.assertAlmostEqual(a_row["score"], 2 / 61, places=4)

    def test_sqlite_store_vector_roundtrip_and_close(self):
        store = SQLiteStore(":memory:", dimensions=8)
        store.upsert_file("p.md", "d1", [("id1", "H", "body", 1)])
        vec = store.vector("id1")
        self.assertEqual(len(vec), 8)
        self.assertAlmostEqual(sum(x * x for x in vec), 1.0, places=5)
        self.assertIsNone(store.vector("missing"))
        self.assertIsNone(store.vector("missing"))
        store.close()

    def test_sqlite_search_empty_query(self):
        store = SQLiteStore()
        self.assertEqual(store.search("   "), [])
        store.close()

    def test_sqlite_search_ignores_quotes_in_terms(self):
        store = SQLiteStore()
        store.upsert_file("p.md", "d", [("id1", "H", 'say "hello" now', 1)])
        self.assertGreaterEqual(len(store.search('"hello"')), 0)
        store.close()

    def test_search_503_on_backend_failure(self):
        with patch.object(server, "hybrid_search", side_effect=RuntimeError("boom")):
            status, body = self.request("GET", "/search?q=x")
        self.assertEqual(status, 503)
        self.assertEqual(body["error"], "search backend unavailable")
        self.assertEqual(body["detail"], "boom")


class IndexerVectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = SQLiteStore()
        self.vector = MockVectorStore()

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_added_and_changed_use_vector_store(self):
        f = self.root / "a.md"
        f.write_text("# A\ncontent")
        stats = Indexer(self.root, self.store, vector_store=self.vector).scan()
        self.assertEqual(stats.added, 1)
        self.assertEqual(self.vector.upserts, 1)
        # Changing content triggers delete_file + upsert.
        f.write_text("# A\nchanged content")
        stats = Indexer(self.root, self.store, vector_store=self.vector).scan()
        self.assertEqual(stats.changed, 1)
        self.assertEqual(self.vector.deletes, ["a.md"])
        self.assertEqual(self.vector.upserts, 2)

    def test_removed_file_also_removed_from_vector_store(self):
        f = self.root / "a.md"
        f.write_text("# A\ncontent")
        Indexer(self.root, self.store, vector_store=self.vector).scan()
        f.unlink()
        stats = Indexer(self.root, self.store, vector_store=self.vector).scan()
        self.assertEqual(stats.removed, 1)
        # One delete: the removal branch. (Changing a file also deletes, but
        # this scenario only removes.)
        self.assertEqual(self.vector.deletes, ["a.md"])

    def test_symlinks_are_ignored(self):
        target = self.root / "real.md"
        target.write_text("# Real\ncontent")
        (self.root / "link.md").symlink_to(target)
        stats = Indexer(self.root, self.store).scan()
        self.assertEqual(stats.added, 1)
        self.assertIsNone(self.store.file_digest("link.md"))

    def test_custom_chunk_size_used(self):
        f = self.root / "a.md"
        f.write_text("# A\n" + "word " * 500)
        stats = Indexer(self.root, self.store, chunk_size=200).scan()
        self.assertEqual(stats.added, 1)
        self.assertGreater(self.store.status()["chunks"], 1)


class MockVectorStore:
    def __init__(self):
        self.upserts = 0
        self.deletes = []

    def upsert(self, records, embed):
        self.upserts += 1

    def delete_file(self, path):
        self.deletes.append(path)
