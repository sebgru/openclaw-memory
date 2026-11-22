"""Public-safe synthetic ranking fixtures only."""

import json
import os
import sys
import unittest
from datetime import date
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from memory_store import server
from memory_store.fusion import (
    canonical_path,
    document_date,
    fuse_scopes,
    metadata_adjustment,
    reciprocal_rank,
    session_date,
    tier_prior,
)
from memory_store.qdrant import point_id


def hit(name, source="memory", **kwargs):
    return {
        "id": name,
        "path": name + ".md",
        "text": "text " + name,
        "source": source,
        "line": 1,
        "heading": "H",
        "score": 10,
        **kwargs,
    }


class RankFusionTests(unittest.TestCase):
    def test_rrf_cross_scope_ignores_raw_scores_and_accumulates_independent_lanes(self):
        a = hit("a", score=0.001)
        b = hit("b", score=1e8)
        results = fuse_scopes([("main", [a, b]), ("archive", [a])])
        self.assertEqual(results[0]["id"], "a")
        self.assertAlmostEqual(results[0]["fusion_debug"]["rrf_score"], 2 / 61)
        self.assertEqual(results[0]["fusion_debug"]["scope_ranks"], {"main": 1, "archive": 1})
        self.assertEqual(results[0]["fusion_debug"]["post_fusion_rank"], 1)
        self.assertEqual(reciprocal_rank(2, 0.5), 0.5 / 62)

    def test_tier_prior_and_document_date_order(self):
        today = date(2026, 11, 22)
        rows = [
            hit("old", "document", doc_date="1998-01-01"),
            hit("new", "document", doc_date="2026-11-01"),
            hit("recent", "archive", doc_date="2026-11-20"),
            hit("USER"),
        ]
        results = fuse_scopes([(str(i), [row]) for i, row in enumerate(rows)], today=today)
        self.assertEqual([r["id"] for r in results], ["USER", "recent", "new", "old"])
        self.assertEqual(tier_prior(hit("recent", "archive", doc_date="2026-11-20"), today), 3)
        self.assertEqual(tier_prior(hit("old", "archive", doc_date="2000-01-01"), today), 2)
        self.assertEqual(tier_prior(hit("s", "session")), 3)
        self.assertEqual(tier_prior(hit("outputs/report", "artifact")), 4)
        self.assertEqual(tier_prior(hit("projects/plan")), 4)

    def test_archive_recent_timestamp_without_ingestion_metadata(self):
        today = date(2026, 11, 22)
        row = hit("archive/session", "archive", heading="user — 2026-11-21 12:00:00 UTC")
        self.assertEqual(tier_prior(row, today), 3)
        self.assertEqual(session_date(hit("sessions/2026-11-20", "archive")), date(2026, 11, 20))
        self.assertIsNone(session_date(hit("undated", "archive", heading="2026-13-99")))
        self.assertEqual(tier_prior(hit("future", "archive", doc_date="2099-01-01"), today), 2)

    def test_embedding_fallback_logs_no_query_or_exception_content(self):
        from memory_store.embeddings import EmbeddingClient

        client = EmbeddingClient("http://localhost", dimensions=8)
        with (
            patch.object(client, "_fetch", side_effect=ValueError("synthetic_secret")),
            patch("memory_store.embeddings.time.sleep"),
            self.assertLogs("memory_store.embeddings", level="WARNING") as logs,
        ):
            client.embed("synthetic_secret")
        self.assertNotIn("synthetic_secret", "\n".join(logs.output))

    def test_copy_path_patterns_and_no_overbroad_normalization(self):
        for path in [
            "startup-backup-2026-01-01/notes/topic.md",
            "notes/topic.md.bak",
            "notes/topic-copy.md",
            "notes/topic (copy 2).md",
            "notes/topic-backup.md",
            "notes/topic.md.bak.bak",
            "copies/notes/topic.md",
        ]:
            self.assertEqual(canonical_path(path), ("notes/topic.md", True))
        self.assertEqual(canonical_path("notes/2026/topic.md"), ("notes/2026/topic.md", False))
        self.assertTrue(canonical_path("startup-memory-trim-2026-01-01.md")[1])

    def test_dedup_keeps_canonical_despite_worse_rank_and_merges_provenance(self):
        copy = hit("notes/topic-copy", text="older wording", provenance={"path": "copy"})
        canonical = hit("notes/topic", text="current wording", provenance={"path": "canonical"})
        results = fuse_scopes([("main", [copy, canonical])])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "notes/topic")
        self.assertEqual(results[0]["text"], "current wording")
        self.assertEqual(results[0]["alternate_provenance"], [{"path": "copy"}])
        self.assertEqual(results[0]["fusion_debug"]["duplicates_collapsed"], 1)
        self.assertEqual(results[0]["fusion_debug"]["rrf_score"], 1 / 61)

    def test_text_and_hash_duplicates_and_transitive_groups(self):
        a = hit("a", content_hash="digest", text="Shared Text")
        b = hit("b", content_hash="digest", text="different")
        c = hit("c", text="  shared   text ")
        results = fuse_scopes([("main", [a, b, c])])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["fusion_debug"]["duplicates_collapsed"], 2)
        # Bridge two previously distinct groups.
        results = fuse_scopes(
            [("main", [hit("a", text="one"), hit("b", text="two"), hit("a", text="two")])]
        )
        self.assertEqual(len(results), 1)

    def test_distinct_document_chunks_are_not_collapsed(self):
        results = fuse_scopes(
            [("main", [hit("one", path="book.md", line=1), hit("two", path="book.md", line=30)])]
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(fuse_scopes([]), [])

    def test_relative_paths_in_distinct_corpora_do_not_merge(self):
        rows = [
            hit("one", path="notes.md", text="workspace note"),
            hit("two", "document", path="notes.md", text="document note"),
        ]
        self.assertEqual(len(fuse_scopes([("main", rows)])), 2)
        rows = [
            hit("one", path="same.md", text="first", heading=None, line=None),
            hit("two", path="same.md", text="second", heading=None, line=None),
        ]
        self.assertEqual(len(fuse_scopes([("main", rows)])), 2)

    def test_unmatched_copy_is_penalized(self):
        copy = hit("startup-memory-trim-2026-01-01")
        normal = hit("canonical")
        results = fuse_scopes([("main", [copy, normal])])
        self.assertEqual(results[0]["id"], "canonical")
        self.assertEqual(results[1]["fusion_debug"]["copy_penalty"], 1)

    def test_optional_metadata_neutral_and_malformed_defaults(self):
        row = hit("doc", "document")
        self.assertEqual(metadata_adjustment(row), 0)
        for value in (None, "invalid", "2026-13-99"):
            self.assertIsNone(document_date({**row, "doc_date": value}))
        for quality in (None, "low", float("nan"), float("inf"), True):
            self.assertEqual(metadata_adjustment({**row, "ocr_quality": quality}), 0)
        self.assertEqual(metadata_adjustment({**row, "metadata": {"ocr_quality": 0}}), -0.25)
        self.assertEqual(metadata_adjustment({**row, "ocr_quality": 1}), 0)
        self.assertEqual(metadata_adjustment({**row, "doc_date": "2099-01-01"}), 0.25)
        self.assertEqual(metadata_adjustment(hit("main", ocr_quality=0)), 0)

    def test_identifier_query_runs_lexical_exact_and_vector_in_rrf_mode(self):
        selected = Mock(spec=["search", "search_exact", "search_fuzzy"])
        selected.search.return_value = [hit("lexical", score=0.1)]
        selected.search_exact.return_value = []
        selected.search_fuzzy.return_value = []
        vector = Mock()
        vector.search.return_value = [hit(point_id("semantic"), score=0.99)]
        with (
            patch.dict(os.environ, {"UNIFIED_FUSION_MODE": "rrf"}),
            patch.object(server, "store", selected),
            patch.object(server, "vector_store", vector),
            patch.object(server.embedder, "embed", return_value=[1]),
            patch.object(server, "nimble_rerank", side_effect=lambda q, rows: rows),
        ):
            results, _ = server.unified_search("sensor.example_temperature_01", 10, "main")
        selected.search.assert_called_once()
        selected.search_exact.assert_called_once()
        vector.search.assert_called_once()
        # Rank-only mode cannot preserve semantic-only raw-score dominance.
        self.assertEqual(results[0]["path"], "lexical.md")
        self.assertEqual(results[1]["semantic_similarity"], 0.99)
        self.assertFalse(getattr(server._search_state, "rank_fusion", False))

    def test_within_scope_rrf_rewards_joint_evidence_not_raw_similarity(self):
        selected = Mock(spec=["search"])
        selected.search.return_value = [hit("joint", score=0.0001)]
        vector = Mock()
        vector.search.return_value = [
            hit(point_id("joint"), score=0.01),
            hit(point_id("semantic"), score=0.99),
        ]
        with (
            patch.object(server._search_state, "rank_fusion", True, create=True),
            patch.object(server.embedder, "embed", return_value=[1]),
        ):
            results = server.hybrid_search("unseen", 5, selected, vector)
        self.assertEqual(results[0]["path"], "joint.md")
        self.assertAlmostEqual(results[0]["score"], 2 / 61)
        self.assertAlmostEqual(results[1]["score"], 1 / 62)
        self.assertEqual(results[0]["semantic_similarity"], 0.01)

    def test_rrf_vector_fallback_disclosed_without_query_in_logs(self):
        selected = Mock(spec=["search"])
        selected.search.return_value = [hit("lexical", score=1)]
        vector = Mock()
        vector.search.side_effect = RuntimeError("private_synthetic_token")
        with (
            patch.dict(os.environ, {"UNIFIED_FUSION_MODE": "rrf"}),
            patch.object(server, "store", selected),
            patch.object(server, "vector_store", vector),
            patch.object(server.embedder, "embed", return_value=[1]),
            self.assertLogs("memory_store", level="WARNING") as logs,
        ):
            results, warnings, coverage, degraded = server.unified_search_with_coverage(
                "private_synthetic_token", 5, "main"
            )
        self.assertEqual(len(results), 1)
        self.assertEqual(coverage["main"], "searched")
        self.assertEqual(degraded, {"main": "lexical_fallback"})
        self.assertIn("fallback", warnings[0])
        self.assertNotIn("private_synthetic_token", "\n".join(logs.output))
        self.assertFalse(getattr(server._search_state, "rank_fusion", False))

    def test_nimble_one_call_cannot_override_final_tier_prior(self):
        main, documents = object(), object()

        def hybrid(query, limit, selected, vector):
            if selected is main:
                return [hit("curated"), hit("main-second")]
            return [hit("doc", "document"), hit("doc-second", "document")]

        with (
            patch.dict(os.environ, {"UNIFIED_FUSION_MODE": "rrf"}),
            patch.object(server, "store", main),
            patch.object(server, "archive_store", None),
            patch.object(server, "documents_store", documents),
            patch.object(server, "hybrid_search", side_effect=hybrid),
            patch.object(
                server, "nimble_rerank", side_effect=lambda q, rows: list(reversed(rows))
            ) as nimble,
        ):
            results, _ = server.unified_search("topic", 10)
        self.assertEqual(
            [r["source"] for r in results], ["memory", "memory", "document", "document"]
        )
        nimble.assert_called_once()
        self.assertEqual(
            [r["path"] for r in nimble.call_args.args[1]],
            ["curated.md", "doc.md", "main-second.md", "doc-second.md"],
        )

    def test_weak_no_match_remains_low_confidence_after_prior(self):
        selected = Mock(spec=["search"])
        selected.search.return_value = []
        vector = Mock()
        vector.search.return_value = [hit("semantic", score=0.2)]
        with (
            patch.dict(os.environ, {"UNIFIED_FUSION_MODE": "rrf"}),
            patch.object(server, "store", selected),
            patch.object(server, "vector_store", vector),
            patch.object(server.embedder, "embed", return_value=[1]),
        ):
            results, _ = server.unified_search("unseen", 5, "main")
        self.assertTrue(server.is_low_confidence(results))
        self.assertGreater(results[0]["score"], 0)

    def test_invalid_config_fails_explicitly(self):
        with patch.dict(os.environ, {"UNIFIED_FUSION_MODE": "mistyped"}):
            with self.assertRaisesRegex(ValueError, "UNIFIED_FUSION_MODE"):
                server.unified_search("x", 1)

    def test_debug_opt_in_and_query_not_logged(self):
        selected = Mock(spec=["search"])
        selected.search.return_value = [hit("a", score=1)]
        http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        Thread(target=http.serve_forever, daemon=True).start()
        try:
            with (
                patch.dict(os.environ, {"UNIFIED_FUSION_MODE": "rrf"}),
                patch.object(server, "store", selected),
                patch.object(server, "vector_store", None),
                self.assertLogs("memory_store", level="INFO") as logs,
            ):
                payloads = []
                for debug in ("0", "1"):
                    connection = HTTPConnection(*http.server_address)
                    connection.request(
                        "GET", "/unified/search?q=private_synthetic_token&scope=main&debug=" + debug
                    )
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    payloads.append(json.loads(response.read()))
                    connection.close()
            self.assertNotIn("fusion_debug", payloads[0]["results"][0])
            self.assertIn("fusion_debug", payloads[1]["results"][0])
            self.assertNotIn("private_synthetic_token", "\n".join(logs.output))
        finally:
            http.shutdown()
            http.server_close()


if __name__ == "__main__":
    unittest.main()
