import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread

from memory_store import server


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        Thread(target=cls.http.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()

    def request(self, path):
        c = HTTPConnection(*self.http.server_address)
        c.request("GET", path)
        r = c.getresponse()
        return r.status

    def request_json(self, path):
        c = HTTPConnection(*self.http.server_address)
        c.request("GET", path)
        r = c.getresponse()
        return r.status, json.loads(r.read())

    def test_validation(self):
        self.assertEqual(self.request("/search"), 400)
        self.assertEqual(self.request("/search?q=x&limit=no"), 400)
        self.assertEqual(self.request("/status"), 200)

    def test_search_exposes_separate_hybrid_scores(self):
        original = server.store.search
        try:
            server.store.search = lambda query, limit: [
                {"id": "one", "text": "fact", "path": "a.md", "score": 2.0}
            ]
            result = server.hybrid_search("fact", 1, server.store, None)[0]
            self.assertGreater(result["lexical_score"], 0)
            self.assertEqual(result["semantic_score"], 0)
            self.assertEqual(result["score"], result["lexical_score"])
        finally:
            server.store.search = original

    def test_search_labels_output_registry_as_artifact(self):
        original = server.store.search
        try:
            server.store.search = lambda query, limit: [
                {"id": "one", "text": "artifact", "path": "outputs/INDEX.md", "score": 2.0}
            ]
            result = server.hybrid_search("artifact", 1, server.store, None)[0]
            self.assertEqual(result["source"], "artifact")
        finally:
            server.store.search = original

    def test_unified_search_merges_backend_ids_and_labels_sources(self):
        original_hybrid = server.hybrid_search
        original_archive = server.archive_store
        try:
            server.archive_store = object()

            def fake_hybrid(query, limit, selected_store, selected_vector_store):
                if selected_store is server.store:
                    return [
                        {
                            "id": "sqlite-id",
                            "path": "outputs/INDEX.md",
                            "heading": "Report",
                            "text": "same result",
                            "line": 3,
                            "score": 0.2,
                            "lexical_score": 0.1,
                            "semantic_score": 0.1,
                        },
                        {
                            "id": "qdrant-id",
                            "path": "outputs/INDEX.md",
                            "heading": "Report",
                            "text": "same result",
                            "line": 3,
                            "score": 0.3,
                            "lexical_score": 0,
                            "semantic_score": 0.3,
                        },
                    ]
                return [
                    {
                        "id": "qdrant-id",
                        "path": "session/one.md",
                        "heading": "Decision",
                        "text": "archive result",
                        "line": 8,
                        "score": 0.4,
                        "lexical_score": 0.4,
                        "semantic_score": 0,
                    }
                ]

            server.hybrid_search = fake_hybrid
            results, warnings = server.unified_search("same", 10)
            self.assertEqual(warnings, [])
            self.assertEqual(len(results), 2)
            self.assertEqual({row["source"] for row in results}, {"artifact", "archive"})
            self.assertTrue(all(len(row["id"]) == 64 for row in results))
            self.assertTrue(
                all("lexical_score" in row and "semantic_score" in row for row in results)
            )
            artifact = next(row for row in results if row["source"] == "artifact")
            self.assertEqual(artifact["score"], 0.5)
            self.assertEqual(artifact["lexical_score"], 0.1)
            self.assertEqual(artifact["semantic_score"], 0.4)
        finally:
            server.hybrid_search = original_hybrid
            server.archive_store = original_archive

    def test_unified_search_reports_partial_backend_failure(self):
        original_hybrid = server.hybrid_search
        original_archive = server.archive_store
        try:
            server.archive_store = object()

            def fake_hybrid(query, limit, selected_store, selected_vector_store):
                if selected_store is server.store:
                    raise OSError("main unavailable")
                return [
                    {
                        "id": "archive-id",
                        "path": "old.md",
                        "heading": "Old",
                        "text": "archived fact",
                        "line": 1,
                        "score": 0.5,
                        "lexical_score": 0.5,
                        "semantic_score": 0,
                    }
                ]

            server.hybrid_search = fake_hybrid
            results, warnings = server.unified_search("fact", 10)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["source"], "archive")
            self.assertEqual(warnings, ["main: search backend unavailable"])
        finally:
            server.hybrid_search = original_hybrid
            server.archive_store = original_archive

    def test_unified_search_rejects_invalid_scope(self):
        with self.assertRaises(ValueError):
            server.unified_search("fact", 10, "sessions")

    def test_unified_search_rejects_archive_without_archive_store(self):
        original_archive = server.archive_store
        try:
            server.archive_store = None
            with self.assertRaises(LookupError):
                server.unified_search("fact", 10, "archive")
        finally:
            server.archive_store = original_archive

    def test_unified_search_main_scope_skips_archive_warning(self):
        original_hybrid = server.hybrid_search
        original_archive = server.archive_store
        try:
            server.archive_store = None

            def fake_hybrid(query, limit, selected_store, selected_vector_store):
                return [
                    {
                        "id": "one",
                        "path": "notes/idea.md",
                        "heading": "Idea",
                        "text": "memory fact",
                        "line": 2,
                        "score": 0.5,
                        "lexical_score": 0.5,
                        "semantic_score": 0,
                    }
                ]

            server.hybrid_search = fake_hybrid
            results, warnings = server.unified_search("fact", 10, "main")
            self.assertEqual(warnings, [])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["source"], "memory")
        finally:
            server.hybrid_search = original_hybrid
            server.archive_store = original_archive

    def test_unified_source_labels_session_paths(self):
        self.assertEqual(server.unified_source({"path": "sessions/abc.md"}), "session")
        self.assertEqual(server.unified_source({"path": "logs/x/sessions/abc.md"}), "session")
        self.assertEqual(server.unified_source({"path": "outputs/INDEX.md"}), "artifact")
        self.assertEqual(server.unified_source({"path": "notes/idea.md"}), "memory")
        self.assertEqual(server.unified_source({"path": "anything.md"}, archive=True), "archive")

    def test_unified_search_all_scope_without_archive_warns(self):
        original_hybrid = server.hybrid_search
        original_archive = server.archive_store
        try:
            server.archive_store = None

            def fake_hybrid(query, limit, selected_store, selected_vector_store):
                return [
                    {
                        "id": "one",
                        "path": "notes/idea.md",
                        "heading": "Idea",
                        "text": "memory fact",
                        "line": 2,
                        "score": 0.5,
                        "lexical_score": 0.5,
                        "semantic_score": 0,
                    }
                ]

            server.hybrid_search = fake_hybrid
            results, warnings = server.unified_search("fact", 10, "all")
            self.assertEqual(warnings, ["archive: not configured"])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["source"], "memory")
        finally:
            server.hybrid_search = original_hybrid
            server.archive_store = original_archive

    def test_unified_search_archive_backend_failure_continues(self):
        original_hybrid = server.hybrid_search
        original_archive = server.archive_store
        try:
            server.archive_store = object()

            def fake_hybrid(query, limit, selected_store, selected_vector_store):
                if selected_store is server.store:
                    return [
                        {
                            "id": "one",
                            "path": "notes/idea.md",
                            "heading": "Idea",
                            "text": "memory fact",
                            "line": 2,
                            "score": 0.5,
                            "lexical_score": 0.5,
                            "semantic_score": 0,
                        }
                    ]
                raise OSError("archive unavailable")

            server.hybrid_search = fake_hybrid
            results, warnings = server.unified_search("fact", 10)
            self.assertEqual(warnings, ["archive: search backend unavailable"])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["source"], "memory")
        finally:
            server.hybrid_search = original_hybrid
            server.archive_store = original_archive

    def test_unified_endpoint_validation_and_errors(self):
        self.assertEqual(self.request("/unified/search"), 400)
        self.assertEqual(self.request("/unified/search?q=fact&limit=no"), 400)
        self.assertEqual(self.request("/unified/search?q=fact&limit=0"), 400)
        self.assertEqual(self.request("/unified/search?q=fact&limit=101"), 400)
        self.assertEqual(self.request("/unified/search?q=fact&scope=bogus"), 400)
        original_archive = server.archive_store
        try:
            server.archive_store = None
            status, payload = self.request_json("/unified/search?q=fact&scope=archive")
            self.assertEqual(status, 404)
            self.assertEqual(payload["error"], "archive is not configured")
        finally:
            server.archive_store = original_archive

    def test_unified_endpoint_omits_warnings_when_empty(self):
        original = server.unified_search
        try:
            server.unified_search = lambda query, limit, scope: (
                [{"id": "stable", "source": "memory", "score": 1}],
                [],
            )
            status, payload = self.request_json("/unified/search?q=fact")
            self.assertEqual(status, 200)
            self.assertNotIn("warnings", payload)
        finally:
            server.unified_search = original

    def test_unified_endpoint_returns_results_and_warnings(self):
        original = server.unified_search
        try:
            server.unified_search = lambda query, limit, scope: (
                [{"id": "stable", "source": "memory", "score": 1}],
                ["archive: search backend unavailable"],
            )
            status, payload = self.request_json("/unified/search?q=fact&scope=main&limit=2")
            self.assertEqual(status, 200)
            self.assertEqual(payload["results"][0]["id"], "stable")
            self.assertEqual(payload["warnings"], ["archive: search backend unavailable"])
        finally:
            server.unified_search = original


if __name__ == "__main__":
    unittest.main()
