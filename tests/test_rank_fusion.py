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
        copy = hit("notes/topic-copy", text="shared wording", provenance={"path": "copy"})
        canonical = hit("notes/topic", text="shared wording", provenance={"path": "canonical"})
        results = fuse_scopes([("main", [copy, canonical])])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "notes/topic")
        self.assertEqual(results[0]["text"], "shared wording")
        self.assertEqual(results[0]["alternate_provenance"], [{"path": "copy"}])
        self.assertEqual(results[0]["fusion_debug"]["duplicates_collapsed"], 1)
        self.assertEqual(results[0]["fusion_debug"]["rrf_score"], 1 / 61)

    def test_hash_duplicates_and_session_text_minimum(self):
        a = hit("a", content_hash="digest")
        b = hit("b", content_hash="digest", text="different")
        self.assertEqual(len(fuse_scopes([("main", [a, b])])), 1)
        text = "A long transcript sentence that establishes evidence of shared conversation. " * 2
        a, b = hit("a", "session", text=text), hit("b", "archive", text=text.upper())
        self.assertEqual(len(fuse_scopes([("main", [a]), ("archive", [b])])), 1)
        a["text"] = b["text"] = "okay"
        self.assertEqual(len(fuse_scopes([("main", [a]), ("archive", [b])])), 2)

    def test_transitive_groups_require_explicit_identity_bridges(self):
        a = hit("a", text="shared", content_hash="chunk-digest")
        b = hit("b", text="shared")
        bridge = hit("bridge", path="b.md", text="shared", content_hash="chunk-digest")
        rows = fuse_scopes([("main", [a, b, bridge])])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fusion_debug"]["duplicates_collapsed"], 2)
        # An unrelated same-text document must not enter the component.
        other = hit("other", "document", text="shared")
        self.assertEqual(len(fuse_scopes([("main", [a, b, bridge]), ("documents", [other])])), 2)

    def test_unknown_confidence_metadata_stays_unknown(self):
        rows = fuse_scopes([("main", [hit("a")])])
        self.assertNotIn("semantic_similarity", rows[0])
        self.assertFalse(server.is_low_confidence(rows))

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
        self.assertEqual(results[1]["fusion_debug"]["copy_penalty"], 3)

    def test_optional_metadata_neutral_and_malformed_defaults(self):
        row = hit("doc", "document")
        self.assertEqual(metadata_adjustment(row), 0)
        for value in (None, "invalid", "2026-13-99"):
            self.assertIsNone(document_date({**row, "doc_date": value}))
        for quality in (None, "low", float("nan"), float("inf"), True):
            self.assertEqual(metadata_adjustment({**row, "ocr_quality": quality}), 0)
        self.assertEqual(metadata_adjustment({**row, "metadata": {"ocr_quality": 0}}), -0.5)
        self.assertAlmostEqual(metadata_adjustment({**row, "ocr_quality": 1}), 0.5)
        self.assertEqual(metadata_adjustment({**row, "doc_date": "2099-01-01"}), 0)
        self.assertEqual(metadata_adjustment(hit("main", ocr_quality=0)), 0)

    def test_identifier_query_runs_lexical_exact_and_vector_in_rrf_mode(self):
        selected = Mock(spec=["search", "search_exact", "search_fuzzy"])
        selected.search.return_value = [hit("lexical", score=0.1)]
        selected.search_exact.return_value = []
        selected.search_fuzzy.return_value = []
        vector = Mock()
        vector.search.return_value = [hit(point_id("semantic"), path="semantic.md", score=0.99)]
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
            hit(point_id("semantic"), path="semantic.md", score=0.99),
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

    def test_nimble_one_call_applies_soft_final_prior(self):
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

    def test_soft_prior_interleaves_documents_and_memory(self):
        memory = [hit(f"m{i}") for i in range(1, 13)]
        rows = fuse_scopes([("main", memory), ("documents", [hit("d", "document")])])
        ids = [r["id"] for r in rows]
        self.assertLess(ids.index("m3"), ids.index("d"))
        self.assertLess(ids.index("d"), ids.index("m6"))
        self.assertLess(ids.index("d"), ids.index("m10"))

    def test_duplicates_merge_best_evidence_and_do_not_merge_distinct_documents(self):
        a = hit(
            "a",
            text="same",
            lexical_evidence=0,
            semantic_similarity=0.1,
            relevance_score=0.1,
            exact_match=False,
        )
        b = hit(
            "a-copy",
            text="same",
            lexical_evidence=8,
            semantic_similarity=0.9,
            relevance_score=0.9,
            exact_match=True,
        )
        row = fuse_scopes([("main", [b, a])])[0]
        self.assertEqual(row["path"], "a.md")
        self.assertEqual(row["lexical_evidence"], 8)
        self.assertEqual(row["semantic_similarity"], 0.9)
        self.assertEqual(row["relevance_score"], 0.9)
        self.assertTrue(row["exact_match"])
        self.assertFalse(server.is_low_confidence([row]))
        self.assertEqual(len(server.apply_search_profile([row], 1, "prompt")), 1)
        documents = [
            hit("invoice-a", "document", text="identical boilerplate"),
            hit("invoice-b", "document", text="identical boilerplate"),
            hit("q3", "document", text="first version"),
            hit("q3-trim", "document", text="different version"),
        ]
        self.assertEqual(len(fuse_scopes([("documents", documents)])), 4)

    def test_metadata_neutral_baselines_and_bounded_effect(self):
        today = date(2026, 11, 22)
        row = hit("doc", "document")
        self.assertEqual(metadata_adjustment({**row, "ocr_quality": 0.9}), 0)
        self.assertEqual(metadata_adjustment({**row, "doc_date": "2025-11-22"}, today), 0)
        self.assertEqual(metadata_adjustment({**row, "doc_date": "1899-01-01"}, today), 0)
        self.assertEqual(metadata_adjustment({**row, "ocr_quality": -1}), 0)
        self.assertEqual(metadata_adjustment({**row, "ocr_quality": 2}), 0)

    def test_deterministic_equal_scores(self):
        a, b = hit("a"), hit("b")
        # Same rank in independent lanes: path tie-break is stable.
        for lanes in [[("x", [a]), ("y", [b])], [("y", [b]), ("x", [a])]] * 3:
            self.assertEqual([r["id"] for r in fuse_scopes(lanes)], ["a", "b"])

    def test_library_debug_is_opt_in(self):
        selected = Mock(spec=["search"])
        selected.search.return_value = [hit("a", score=1)]
        with (
            patch.dict(os.environ, {"UNIFIED_FUSION_MODE": "rrf"}),
            patch.object(server, "store", selected),
            patch.object(server, "vector_store", None),
        ):
            rows, _ = server.unified_search("unseen", 1, "main")
            self.assertNotIn("fusion_debug", rows[0])
            rows, *_ = server.unified_search_with_coverage("unseen", 1, "main", debug=True)
            self.assertEqual(rows[0]["fusion_debug"]["post_profile_rank"], 1)
            self.assertEqual(rows[0]["fusion_debug"]["engine_ranks"], [{"lexical": 1}])

    def test_nimble_receives_canonical_candidates_only(self):
        rows = [hit("a-copy", text="shared"), hit("a", text="shared"), hit("b")]
        with (
            patch.dict(os.environ, {"UNIFIED_FUSION_MODE": "rrf"}),
            patch.object(server, "hybrid_search", return_value=rows),
            patch.object(server, "nimble_rerank", side_effect=lambda q, rows: rows) as nimble,
        ):
            server.unified_search("topic", 10, "main")
        self.assertEqual([r["path"] for r in nimble.call_args.args[1]], ["a.md", "b.md"])

    def test_http_confidence_before_truncation_and_invalid_mode(self):
        main, documents = object(), object()

        def hybrid(query, limit, selected, vector):
            if selected is main:
                return [hit("weak", semantic_similarity=0.1, lexical_evidence=0)]
            return [hit("strong", "document", semantic_similarity=0.1, lexical_evidence=2)]

        http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        Thread(target=http.serve_forever, daemon=True).start()
        try:
            with (
                patch.object(server, "store", main),
                patch.object(server, "documents_store", documents),
                patch.object(server, "archive_store", None),
                patch.object(server, "hybrid_search", side_effect=hybrid),
            ):
                for mode in ("rrf", "legacy", "mistyped"):
                    with patch.dict(os.environ, {"UNIFIED_FUSION_MODE": mode}):
                        connection = HTTPConnection(*http.server_address)
                        connection.request("GET", "/unified/search?q=topic&limit=1&debug=1")
                        response = connection.getresponse()
                        payload = json.loads(response.read())
                        connection.close()
                    self.assertEqual(response.status, 500 if mode == "mistyped" else 200)
                    if mode != "mistyped":
                        self.assertEqual(payload.get("low_confidence", False), mode == "legacy")
                        self.assertEqual(payload["fusion"], mode)
                        if mode == "legacy":
                            self.assertNotIn("fusion_debug", payload["results"][0])
        finally:
            http.shutdown()
            http.server_close()

    def test_legacy_default_confidence_matches_main_7f6e316_delivered_rows(self):
        # Main 7f6e316 evaluates the final HTTP results, not hidden candidates.
        # Pin its predicate independently so a shared helper regression is visible.
        def main_low_confidence(rows):
            return bool(rows) and all(
                "semantic_similarity" in row
                and row.get("lexical_evidence", 0) <= 0
                and not row.get("exact_match")
                and row["semantic_similarity"] < server.LOW_CONFIDENCE_SEMANTIC_FLOOR
                for row in rows
            )

        weak = hit("weak", semantic_similarity=0.1, lexical_evidence=0)
        strong = hit("strong", semantic_similarity=0.9, lexical_evidence=2)
        cases = (
            ([weak, strong], "tool", True),
            ([weak, strong], "prompt", False),  # profile filters both rows
            ([], "tool", False),
            ([hit("unknown")], "tool", True),  # legacy merge fills missing evidence with zero
            ([{**weak, "exact_match": True}], "tool", False),
            (
                [{**weak, "semantic_similarity": server.LOW_CONFIDENCE_SEMANTIC_FLOOR}],
                "tool",
                False,
            ),
        )
        http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        Thread(target=http.serve_forever, daemon=True).start()
        try:
            for mode in (None, "legacy"):
                for rows, profile, expected in cases:
                    with (
                        self.subTest(mode=mode, profile=profile, rows=rows),
                        patch.dict(os.environ),
                        patch.object(server, "hybrid_search", return_value=rows),
                        patch.object(server, "nimble_rerank", side_effect=lambda q, rows: rows),
                    ):
                        if mode is None:
                            os.environ.pop("UNIFIED_FUSION_MODE", None)
                        else:
                            os.environ["UNIFIED_FUSION_MODE"] = mode
                        connection = HTTPConnection(*http.server_address)
                        connection.request(
                            "GET", f"/unified/search?q=topic&scope=main&limit=1&profile={profile}"
                        )
                        response = connection.getresponse()
                        payload = json.loads(response.read())
                        connection.close()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(main_low_confidence(payload["results"]), expected)
                        self.assertEqual(payload.get("low_confidence", False), expected)
        finally:
            http.shutdown()
            http.server_close()

    def test_noop_nimble_preserves_deep_cross_lane_duplicate_ranks(self):
        text = "Synthetic shared session evidence for cross-lane rank preservation. " * 3
        main, archive = object(), object()
        duplicate = hit("sessions/shared", text=text)
        archived = hit("copies/shared", text=text)
        for main_rank, archive_rank in ((50, 1), (1, 8)):
            main_rows = [hit(f"m{i}") for i in range(1, main_rank)] + [duplicate]
            archive_rows = [hit(f"a{i}") for i in range(1, archive_rank)] + [archived]

            def hybrid(
                query, limit, selected, vector, main_rows=main_rows, archive_rows=archive_rows
            ):
                return main_rows if selected is main else archive_rows

            with (
                patch.dict(os.environ, {"UNIFIED_FUSION_MODE": "rrf", "NIMBLE_RERANK_URL": ""}),
                patch.object(server, "store", main),
                patch.object(server, "archive_store", archive),
                patch.object(server, "documents_store", None),
                patch.object(server, "hybrid_search", side_effect=hybrid),
            ):
                rows, *_ = server.unified_search_with_coverage("synthetic", 100, debug=True)
            shared = next(row for row in rows if row["path"] == duplicate["path"])
            self.assertEqual(
                shared["fusion_debug"]["scope_ranks"], {"main": main_rank, "archive": archive_rank}
            )
            self.assertAlmostEqual(shared["score"], 1 / (61 + main_rank) + 1 / (61 + archive_rank))

    def test_other_lane_rerank_cannot_promote_deep_duplicate(self):
        text = "Synthetic cross-lane duplicate whose main rank must remain deep. " * 3
        main, archive = object(), object()
        main_rows = [hit(f"m{i}") for i in range(49)] + [hit("sessions/shared", text=text)]
        archive_rows = [hit("a1"), hit("copies/shared", text=text), hit("a3")]

        def rerank(query, rows):
            rows = list(rows)
            positions = [i for i, row in enumerate(rows) if row["path"] in {"a1.md", "a3.md"}]
            rows[positions[0]], rows[positions[1]] = rows[positions[1]], rows[positions[0]]
            return rows

        with (
            patch.dict(os.environ, {"UNIFIED_FUSION_MODE": "rrf"}),
            patch.object(server, "store", main),
            patch.object(server, "archive_store", archive),
            patch.object(server, "documents_store", None),
            patch.object(
                server,
                "hybrid_search",
                side_effect=lambda q, n, selected, v: (
                    main_rows if selected is main else archive_rows
                ),
            ),
            patch.object(server, "nimble_rerank", side_effect=rerank),
        ):
            rows, *_ = server.unified_search_with_coverage("synthetic", 100, debug=True)
        shared = next(row for row in rows if row["path"] == "sessions/shared.md")
        self.assertEqual(shared["fusion_debug"]["scope_ranks"], {"main": 50, "archive": 2})
        first_archive = next(row for row in rows if row["path"] == "a3.md")
        self.assertEqual(first_archive["fusion_debug"]["scope_ranks"], {"archive": 1})

    def test_rerank_explicitly_maps_transitive_group_members(self):
        # a and b connect only through bridge; b has no key in canonical a.
        rows = [
            hit("a", text="shared", content_hash="digest"),
            hit("middle"),
            hit("b", text="shared"),
            hit("bridge", path="b.md", text="shared", content_hash="digest"),
        ]
        with (
            patch.dict(os.environ, {"UNIFIED_FUSION_MODE": "rrf"}),
            patch.object(server, "hybrid_search", return_value=rows),
            patch.object(server, "nimble_rerank", side_effect=lambda q, rows: list(reversed(rows))),
        ):
            results, *_ = server.unified_search_with_coverage("synthetic", 100, "main", debug=True)
        group = next(row for row in results if row["path"] == "a.md")
        self.assertEqual(group["fusion_debug"]["scope_ranks"], {"main": 2})
        self.assertEqual(group["fusion_debug"]["duplicates_collapsed"], 2)

    def test_full_depth_exact_document_exposes_evidence_blind_prior_limit(self):
        # Deliberate known design limit, not an expected relevance outcome.
        main = [hit(f"m{i}", semantic_similarity=0.1) for i in range(300)]
        archive = [hit(f"a{i}", "archive", doc_date="2026-11-21") for i in range(300)]
        documents = [hit("exact", "document", exact_match=True, lexical_evidence=10)]
        rows = fuse_scopes(
            [("main", main), ("archive", archive), ("documents", documents)],
            today=date(2026, 11, 22),
        )
        self.assertEqual(next(i for i, row in enumerate(rows, 1) if row["id"] == "exact"), 10)

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
