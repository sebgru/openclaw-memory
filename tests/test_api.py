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
            self.assertEqual(result["lexical_evidence"], 2.0)
            self.assertEqual(result["relevance_score"], 1.0)
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
            self.assertEqual(warnings, ["documents: not configured"])
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
            self.assertEqual(
                warnings, ["documents: not configured", "main: search backend unavailable"]
            )
        finally:
            server.hybrid_search = original_hybrid
            server.archive_store = original_archive

    def test_unified_search_deduplicates_repeated_session_content(self):
        original_hybrid = server.hybrid_search
        original_archive = server.archive_store
        try:
            server.archive_store = object()

            def fake_hybrid(query, limit, selected_store, selected_vector_store):
                path = "sessions/live.md" if selected_store is server.store else "copies/old.md"
                return [
                    {
                        "id": path,
                        "path": path,
                        "heading": "Decision",
                        "text": "  Keep   the approved architecture. ",
                        "line": 4,
                        "score": 0.3,
                        "lexical_score": 0.1,
                        "semantic_score": 0.2,
                        "lexical_evidence": 2.0,
                        "semantic_similarity": 0.8,
                        "relevance_score": 0.8,
                    }
                ]

            server.hybrid_search = fake_hybrid
            results, warnings = server.unified_search("architecture", 10)
            self.assertEqual(warnings, ["documents: not configured"])
            self.assertEqual(len(results), 1)
            self.assertEqual(len(results[0]["alternate_provenance"]), 1)
            self.assertEqual(results[0]["score"], 0.3)
        finally:
            server.hybrid_search = original_hybrid
            server.archive_store = original_archive

    def test_prompt_profile_applies_relevance_and_source_quotas(self):
        original_min = server.PROMPT_MIN_RELEVANCE
        original_quotas = server.PROMPT_SOURCE_QUOTAS
        try:
            server.PROMPT_MIN_RELEVANCE = 0.5
            server.PROMPT_SOURCE_QUOTAS = {"memory": 1, "artifact": 1, "session": 1, "archive": 1}
            rows = [
                {"source": "memory", "relevance_score": 0.9, "id": "m1"},
                {"source": "memory", "relevance_score": 0.8, "id": "m2"},
                {"source": "artifact", "relevance_score": 0.7, "id": "a1"},
                {"source": "archive", "relevance_score": 0.4, "id": "old"},
            ]
            selected = server.apply_search_profile(rows, 10, "prompt")
            self.assertEqual([row["id"] for row in selected], ["m1", "a1"])
            self.assertEqual(server.apply_search_profile(rows, 10, "tool"), rows)
        finally:
            server.PROMPT_MIN_RELEVANCE = original_min
            server.PROMPT_SOURCE_QUOTAS = original_quotas

    def test_prompt_profile_stops_when_limit_reached(self):
        original_min = server.PROMPT_MIN_RELEVANCE
        original_quotas = server.PROMPT_SOURCE_QUOTAS
        try:
            server.PROMPT_MIN_RELEVANCE = 0.0
            server.PROMPT_SOURCE_QUOTAS = {
                "memory": 10,
                "artifact": 10,
                "session": 10,
                "archive": 10,
            }
            rows = [{"source": "memory", "relevance_score": 0.9, "id": f"m{i}"} for i in range(5)]
            selected = server.apply_search_profile(rows, 2, "prompt")
            self.assertEqual([row["id"] for row in selected], ["m0", "m1"])
        finally:
            server.PROMPT_MIN_RELEVANCE = original_min
            server.PROMPT_SOURCE_QUOTAS = original_quotas

    def test_unified_search_rejects_invalid_profile(self):
        with self.assertRaises(ValueError):
            server.unified_search("fact", 10, profile="bogus")

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
            self.assertEqual(warnings, ["archive: not configured", "documents: not configured"])
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
            self.assertEqual(
                warnings, ["documents: not configured", "archive: search backend unavailable"]
            )
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
        self.assertEqual(self.request("/unified/search?q=fact&profile=bogus"), 400)
        original_archive = server.archive_store
        try:
            server.archive_store = None
            status, payload = self.request_json("/unified/search?q=fact&scope=archive")
            self.assertEqual(status, 404)
            self.assertEqual(payload["error"], "archive is not configured")
        finally:
            server.archive_store = original_archive

    def test_unified_endpoint_omits_warnings_when_empty(self):
        original = server.unified_search_with_coverage
        try:
            server.unified_search_with_coverage = lambda query, limit, scope: (
                [{"id": "stable", "source": "memory", "score": 1}],
                [],
                {"main": "searched", "archive": "not_searched", "documents": "not_searched"},
                {},
            )
            status, payload = self.request_json("/unified/search?q=fact")
            self.assertEqual(status, 200)
            self.assertNotIn("warnings", payload)
            self.assertNotIn("degraded", payload)
            self.assertEqual(payload["coverage"]["main"], "searched")
        finally:
            server.unified_search_with_coverage = original

    def test_unified_endpoint_returns_results_and_warnings(self):
        original = server.unified_search_with_coverage
        try:
            server.unified_search_with_coverage = lambda query, limit, scope: (
                [{"id": "stable", "source": "memory", "score": 1}],
                ["archive: search backend unavailable"],
                {"main": "searched", "archive": "unavailable", "documents": "not_searched"},
                {},
            )
            status, payload = self.request_json("/unified/search?q=fact&scope=main&limit=2")
            self.assertEqual(status, 200)
            self.assertEqual(payload["results"][0]["id"], "stable")
            self.assertEqual(payload["warnings"], ["archive: search backend unavailable"])
        finally:
            server.unified_search_with_coverage = original

    def _coverage_fixture(self, failing=()):
        class Store:
            def __init__(self, name, fail):
                self.name, self.fail, self.calls = name, fail, 0

            def search(self, query, limit):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("boom")
                return [{"id": self.name, "text": "fact", "path": f"{self.name}.md", "score": 1.0}]

        return {n: Store(n, n in failing) for n in ("main", "archive", "documents")}

    def _with_stores(self, stores, archive=True, documents=True):
        saved = (
            server.store,
            server.archive_store,
            server.documents_store,
            server.vector_store,
            server.archive_vector_store,
            server.documents_vector_store,
        )
        server.store = stores["main"]
        server.archive_store = stores["archive"] if archive else None
        server.documents_store = stores["documents"] if documents else None
        server.vector_store = server.archive_vector_store = server.documents_vector_store = None
        return saved

    def _restore(self, saved):
        (
            server.store,
            server.archive_store,
            server.documents_store,
            server.vector_store,
            server.archive_vector_store,
            server.documents_vector_store,
        ) = saved

    def test_coverage_full(self):
        stores = self._coverage_fixture()
        saved = self._with_stores(stores)
        try:
            status, payload = self.request_json("/unified/search?q=fact&scope=all")
        finally:
            self._restore(saved)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["coverage"],
            {"main": "searched", "archive": "searched", "documents": "searched"},
        )
        self.assertNotIn("warnings", payload)

    def test_coverage_partial_unconfigured_backends(self):
        stores = self._coverage_fixture()
        saved = self._with_stores(stores, archive=False, documents=False)
        try:
            status, payload = self.request_json("/unified/search?q=fact&scope=all")
        finally:
            self._restore(saved)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["coverage"],
            {"main": "searched", "archive": "unavailable", "documents": "unavailable"},
        )
        self.assertEqual(
            payload["warnings"], ["archive: not configured", "documents: not configured"]
        )

    def test_coverage_failed_backend_is_unavailable_despite_200(self):
        stores = self._coverage_fixture(failing=("archive",))
        saved = self._with_stores(stores)
        try:
            status, payload = self.request_json("/unified/search?q=fact&scope=all")
        finally:
            self._restore(saved)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["coverage"],
            {"main": "searched", "archive": "unavailable", "documents": "searched"},
        )
        self.assertEqual(payload["warnings"], ["archive: search backend unavailable"])

    def test_coverage_not_requested_backends_are_not_run(self):
        stores = self._coverage_fixture()
        saved = self._with_stores(stores)
        try:
            status, payload = self.request_json("/unified/search?q=fact&scope=main")
        finally:
            self._restore(saved)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["coverage"],
            {"main": "searched", "archive": "not_searched", "documents": "not_searched"},
        )
        self.assertEqual(
            (stores["main"].calls, stores["archive"].calls, stores["documents"].calls), (1, 0, 0)
        )

    def test_coverage_unconfigured_documents_scope_stays_404(self):
        stores = self._coverage_fixture()
        saved = self._with_stores(stores, documents=False)
        try:
            status, _ = self.request_json("/unified/search?q=fact&scope=documents")
        finally:
            self._restore(saved)
        self.assertEqual(status, 404)

    def test_unified_semantic_failure_warns_lexical_fallback(self):
        class FailingVectors:
            def search(self, vector, limit):
                raise RuntimeError("qdrant down")

        stores = self._coverage_fixture()
        saved = self._with_stores(stores)
        server.vector_store = FailingVectors()
        try:
            status, payload = self.request_json("/unified/search?q=fact&scope=all")
        finally:
            self._restore(saved)
        self.assertEqual(status, 200)
        self.assertEqual(
            payload["coverage"],
            {"main": "searched", "archive": "searched", "documents": "searched"},
        )
        self.assertEqual(
            payload["warnings"],
            ["main: semantic search unavailable, lexical (FTS) fallback used"],
        )
        self.assertEqual(payload["degraded"], {"main": "lexical_fallback"})
        self.assertTrue(payload["results"])

    def test_unified_fallback_flag_does_not_leak_between_backends(self):
        class FailingVectors:
            def search(self, vector, limit):
                raise RuntimeError("qdrant down")

        stores = self._coverage_fixture()
        saved = self._with_stores(stores)
        server.archive_vector_store = FailingVectors()
        try:
            _, payload = self.request_json("/unified/search?q=fact&scope=all")
        finally:
            self._restore(saved)
        self.assertEqual(
            payload["warnings"],
            ["archive: semantic search unavailable, lexical (FTS) fallback used"],
        )
        self.assertEqual(payload["degraded"], {"archive": "lexical_fallback"})


if __name__ == "__main__":
    unittest.main()
