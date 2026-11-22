"""Migration, indexing, header-flag and search-exposure tests for document metadata."""

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from memory_store import server
from memory_store.chunker import chunk_markdown
from memory_store.doc_metadata import METADATA_VERSION
from memory_store.document_indexer import DocumentIndexer
from memory_store.sqlite_store import DOC_METADATA_COLUMNS, SQLiteStore

INVOICE = (
    "# Stromabrechnung\n\nDatum: 12. März 2024\n\n"
    "Sehr geehrte Frau Beispiel, die Nebenkosten für die Wohnung und der Strom wurden "
    "abgerechnet. Bitte prüfen Sie die Zahlung der Rechnung.\n\n"
    "## Details\n\nDer Verbrauch betrug 1200 kWh.\n"
)
LETTER = "# Brief\n\nDear Mr Example, thank you for your letter. We will reply with the notice.\n"


class RecordingEmbed:
    def __init__(self):
        self.inputs = []

    def __call__(self, text):
        self.inputs.append(text)
        return [0.25] * 8


class RecordingVectors:
    def __init__(self):
        self.records = []

    def upsert_precomputed(self, records):
        self.records.extend(records)

    def delete_ids(self, ids):
        pass

    def delete_file(self, path):
        pass


class OldSchemaMigrationTests(unittest.TestCase):
    """A database created before the metadata columns existed keeps all data."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "documents.db"
        db = sqlite3.connect(self.path)
        db.executescript(
            """
            CREATE TABLE files(path TEXT PRIMARY KEY, digest TEXT NOT NULL);
            CREATE TABLE chunks(id TEXT PRIMARY KEY, path TEXT NOT NULL, heading TEXT,
                                body TEXT NOT NULL, line INTEGER, vector BLOB);
            CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED, path, heading, body);
            CREATE TABLE index_metadata(key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO files VALUES ('old/a.md', 'digest-a');
            INSERT INTO chunks VALUES ('c1', 'old/a.md', 'Head', 'legacy needle text', 3, '0.5,0.5');
            INSERT INTO chunks_fts VALUES ('c1', 'old/a.md', 'Head', 'legacy needle text');
            INSERT INTO index_metadata VALUES ('last_index_completed_at', '2026-01-01T00:00:00+00:00');
            """
        )
        db.commit()
        db.close()

    def tearDown(self):
        self.tmp.cleanup()

    def columns(self):
        with sqlite3.connect(self.path) as db:
            return [row[1] for row in db.execute("PRAGMA table_info(chunks)")]

    def test_migration_adds_columns_and_keeps_every_row(self):
        self.assertNotIn("category", self.columns())
        store = SQLiteStore(self.path, 2, enable_fuzzy=False, doc_metadata=True)
        try:
            for name, _ in DOC_METADATA_COLUMNS:
                self.assertIn(name, self.columns())
            row = store.db.execute("SELECT * FROM chunks WHERE id='c1'").fetchone()
            self.assertEqual(
                (row["path"], row["heading"], row["body"], row["line"], row["vector"]),
                ("old/a.md", "Head", "legacy needle text", 3, "0.5,0.5"),
            )
            self.assertEqual(
                [row["doc_date"], row["category"], row["ocr_quality"], row["language"]],
                [None] * 4,
            )
            self.assertEqual(store.file_digest("old/a.md"), "digest-a")
            self.assertEqual(store.index_metadata()["last_index_completed_at"][:4], "2026")
            hit = store.search("needle")[0]
            self.assertEqual(hit["text"], "legacy needle text")
            index = store.db.execute("PRAGMA index_list(chunks)").fetchall()
            self.assertIn("idx_chunks_path", [item[1] for item in index])
        finally:
            store.close()

    def test_migration_is_idempotent_and_legacy_rows_get_no_metadata_keys(self):
        for _ in range(2):
            store = SQLiteStore(self.path, 2, enable_fuzzy=False, doc_metadata=True)
            try:
                results = store.attach_metadata(store.search("needle"))
                self.assertEqual(
                    set(results[0]), {"id", "path", "heading", "text", "line", "score"}
                )
            finally:
                store.close()
        self.assertEqual(self.columns().count("category"), 1)

    def test_store_without_flag_leaves_schema_untouched(self):
        store = SQLiteStore(self.path, 2, enable_fuzzy=False)
        try:
            self.assertEqual(store.attach_metadata([{"path": "x"}]), [{"path": "x"}])
            store.upsert_file_precomputed(
                "n.md", "d", [("c2", "H", "body", 1, [0.0])], {"c2": {"category": "home"}}
            )
        finally:
            store.close()
        self.assertNotIn("category", self.columns())

    def test_new_database_gets_same_schema(self):
        fresh = Path(self.tmp.name) / "fresh.db"
        store = SQLiteStore(fresh, 2, doc_metadata=True)
        store.close()
        with sqlite3.connect(fresh) as db:
            names = [row[1] for row in db.execute("PRAGMA table_info(chunks)")]
        self.assertEqual(names[-4:], [name for name, _ in DOC_METADATA_COLUMNS])

    def test_indexer_works_on_migrated_database(self):
        store = SQLiteStore(self.path, 8, enable_fuzzy=False, doc_metadata=True)
        root = Path(self.tmp.name) / "docs"
        root.mkdir()
        (root / "new.md").write_text(INVOICE)
        try:
            DocumentIndexer(root, store, None, RecordingEmbed()).scan()
            fresh = store.db.execute(
                "SELECT category FROM chunks WHERE path='new.md' LIMIT 1"
            ).fetchone()
            self.assertEqual(fresh["category"], "home")
            self.assertIsNone(store.db.execute("SELECT 1 FROM chunks WHERE id='c1'").fetchone())
        finally:
            store.close()


class IndexingMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "converted"
        (self.root / "personal" / "wohnung").mkdir(parents=True)
        (self.root / "personal" / "wohnung" / "strom.md").write_text(INVOICE)
        (self.root / "letter.md").write_text(LETTER)
        self.db_path = Path(self.tmp.name) / "documents.db"
        self.store = SQLiteStore(self.db_path, 8, enable_fuzzy=False, doc_metadata=True)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def indexer(self, vectors=None, embed=None, **kwargs):
        return DocumentIndexer(self.root, self.store, vectors, embed or RecordingEmbed(), **kwargs)

    def chunk_rows(self, path):
        return self.store.db.execute(
            "SELECT * FROM chunks WHERE path=? ORDER BY line", (path,)
        ).fetchall()

    def test_metadata_is_stored_with_every_chunk(self):
        self.indexer().scan()
        rows = self.chunk_rows("personal/wohnung/strom.md")
        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["doc_date"], "2024-03-12")
            self.assertEqual(row["category"], "home")
            self.assertEqual(row["language"], "de")
            self.assertTrue(0.0 <= row["ocr_quality"] <= 1.0)
        letter = self.chunk_rows("letter.md")[0]
        self.assertIsNone(letter["doc_date"])
        self.assertEqual((letter["category"], letter["language"]), ("other", "en"))

    def test_garbled_chunk_gets_lower_quality_than_clean_chunk(self):
        garbled = (
            "Rcchnung Nr. 4S6 l2.O3.2O24 Zah1ung ~~ ¦¦ ■■ Slr0m 1ll1 ,,,, ;;;; d1e W0hnung ###"
        )
        (self.root / "scan.md").write_text(f"# Scan\n\n{garbled}\n\n## Clean\n\n{LETTER}")
        self.indexer().scan()
        rows = {row["heading"]: row["ocr_quality"] for row in self.chunk_rows("scan.md")}
        self.assertLess(rows["Scan"], 0.4)
        self.assertGreater(rows["Brief"], 0.8)

    def test_flag_off_leaves_ranking_inputs_unchanged(self):
        embed, vectors = RecordingEmbed(), RecordingVectors()
        self.indexer(vectors, embed).scan()
        expected_inputs, expected_chunks = [], {}
        for rel in ("letter.md", "personal/wohnung/strom.md"):
            content = (self.root / rel).read_text()
            for number, chunk in enumerate(chunk_markdown(content, 1600)):
                chunk_id = hashlib.sha256(f"{rel}\0{number}\0{chunk.text}".encode()).hexdigest()
                expected_inputs.append(f"{chunk.heading} {chunk.text}")
                expected_chunks[chunk_id] = (rel, chunk.heading, chunk.text, chunk.start_line)
        self.assertEqual(sorted(embed.inputs), sorted(expected_inputs))
        self.assertEqual({r[0]: (r[1], r[2], r[3], r[4]) for r in vectors.records}, expected_chunks)
        fts = {
            row[0]: (row[1], row[2], row[3])
            for row in self.store.db.execute("SELECT id, path, heading, body FROM chunks_fts")
        }
        self.assertEqual(
            fts,
            {cid: (rel, heading, body) for cid, (rel, heading, body, _) in expected_chunks.items()},
        )
        self.assertEqual(self.store.search("Stromabrechnung")[0]["heading"], "Stromabrechnung")
        self.assertEqual(self.store.search("2024-03-12"), [])
        self.assertEqual(self.store.search("home"), [])

    def test_flag_defaults_to_off(self):
        self.assertFalse(self.indexer().context_header)
        self.assertEqual(self.indexer().profile, f"meta{METADATA_VERSION}|header=0")

    def test_flag_on_prepends_header_to_embedding_and_fts_only(self):
        embed, vectors = RecordingEmbed(), RecordingVectors()
        self.indexer(vectors, embed, context_header=True).scan()
        header = "Stromabrechnung | 2024-03-12 | home"
        self.assertTrue(any(text.startswith(f"{header}\n") for text in embed.inputs))
        self.assertTrue(all("\n" in text for text in embed.inputs))
        self.assertIn("Brief\n", "".join(embed.inputs))
        strom_inputs = [t for t in embed.inputs if t.startswith(header)]
        self.assertEqual(len(strom_inputs), 2)
        # stored text, ids and vector payload are unchanged
        for record in vectors.records:
            self.assertNotIn(header, record[3])
            self.assertNotIn(header, record[2])
        for row in self.chunk_rows("personal/wohnung/strom.md"):
            self.assertNotIn("2024-03-12", row["body"] + (row["heading"] or ""))
        # but the lexical index now matches header terms
        hits = self.store.search("2024-03-12")
        self.assertEqual({h["path"] for h in hits}, {"personal/wohnung/strom.md"})
        self.assertNotIn("2024-03-12", hits[0]["text"] + hits[0]["heading"])
        self.assertEqual(self.store.search("home")[0]["path"], "personal/wohnung/strom.md")
        self.assertEqual(self.store.search("Brief")[0]["path"], "letter.md")

    def test_chunk_ids_do_not_depend_on_flag(self):
        self.indexer().scan()
        before = {row["id"] for row in self.chunk_rows("personal/wohnung/strom.md")}
        self.indexer(context_header=True).scan(reindex=True)
        after = {row["id"] for row in self.chunk_rows("personal/wohnung/strom.md")}
        self.assertEqual(before, after)

    def test_plain_scan_never_reindexes_unchanged_files(self):
        self.indexer().scan()
        embed = RecordingEmbed()
        stats = self.indexer(embed=embed, context_header=True).scan()
        self.assertEqual((stats.added, stats.changed, stats.unchanged), (0, 0, 2))
        self.assertEqual(embed.inputs, [])

    def test_reindex_redoes_files_indexed_under_another_profile_and_is_resumable(self):
        self.indexer().scan()
        self.assertEqual(
            set(self.store.db.execute("SELECT profile FROM document_index_profile").fetchall()[0]),
            {"meta1|header=0"},
        )
        embed = RecordingEmbed()
        indexer = self.indexer(embed=embed, context_header=True)
        first = indexer.scan(max_files=1, reindex=True)
        self.assertEqual((first.changed, first.pending), (1, 1))
        second = indexer.scan(max_files=1, reindex=True)
        self.assertEqual((second.changed, second.pending), (1, 0))
        third = indexer.scan(reindex=True)
        self.assertEqual((third.changed, third.unchanged, third.pending), (0, 2, 0))
        profiles = {
            r[0] for r in self.store.db.execute("SELECT profile FROM document_index_profile")
        }
        self.assertEqual(profiles, {"meta1|header=1"})
        self.assertTrue(all("\n" in text for text in embed.inputs))
        self.assertIn("Brief\n", "".join(embed.inputs))

    def test_reindex_backfills_metadata_for_legacy_chunks_and_skips_nothing_without_profile(self):
        self.indexer().scan()
        with self.store.db:
            self.store.db.execute("DELETE FROM document_index_profile")
            self.store.db.execute(
                "UPDATE chunks SET doc_date=NULL, category=NULL, ocr_quality=NULL, language=NULL"
            )
        self.assertEqual(self.indexer().scan().unchanged, 2)
        self.assertIsNone(self.chunk_rows("letter.md")[0]["category"])
        stats = self.indexer().scan(reindex=True)
        self.assertEqual(stats.changed, 2)
        self.assertEqual(self.chunk_rows("letter.md")[0]["category"], "other")

    def test_removed_file_drops_profile_row(self):
        indexer = self.indexer()
        indexer.scan()
        (self.root / "letter.md").unlink()
        indexer.scan()
        paths = {r[0] for r in self.store.db.execute("SELECT path FROM document_index_profile")}
        self.assertEqual(paths, {"personal/wohnung/strom.md"})

    def test_attach_metadata_exposes_fields_for_lexical_and_semantic_hits(self):
        self.indexer().scan()
        hit = self.store.search("Verbrauch")[0]
        enriched = self.store.attach_metadata([hit, {"path": "gone.md", "text": "x"}])
        self.assertEqual(enriched[0]["doc_date"], "2024-03-12")
        self.assertEqual(enriched[0]["category"], "home")
        self.assertEqual(enriched[0]["language"], "de")
        self.assertIn("ocr_quality", enriched[0])
        self.assertNotIn("category", enriched[1])
        self.assertEqual(self.store.attach_metadata([]), [])
        semantic_only = {"path": hit["path"], "text": hit["text"], "line": hit["line"]}
        self.assertEqual(self.store.attach_metadata([semantic_only])[0]["category"], "home")


class MetadataApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "converted"
        cls.root.mkdir()
        (cls.root / "strom.md").write_text(INVOICE)
        cls.store = SQLiteStore(
            Path(cls.tmp.name) / "documents.db", 8, enable_fuzzy=False, doc_metadata=True
        )
        cls.names = (
            "DOCUMENTS_ROOT",
            "documents_store",
            "documents_vector_store",
            "DOCUMENTS_INCLUDE_PATTERNS",
            "DOCUMENTS_CONTEXT_HEADER",
        )
        cls.previous = tuple(getattr(server, name) for name in cls.names)
        server.DOCUMENTS_ROOT = str(cls.root)
        server.documents_store = cls.store
        server.documents_vector_store = None
        server.DOCUMENTS_INCLUDE_PATTERNS = ("**/*.md",)
        server.DOCUMENTS_CONTEXT_HEADER = False
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        Thread(target=cls.http.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.store.close()
        for name, value in zip(cls.names, cls.previous, strict=True):
            setattr(server, name, value)
        cls.tmp.cleanup()

    def request(self, method, path, payload=None):
        connection = HTTPConnection(*self.http.server_address)
        connection.request(method, path, body=None if payload is None else json.dumps(payload))
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def test_search_results_expose_metadata(self):
        self.assertEqual(self.request("POST", "/documents/index", {"max_files": 10})[0], 200)
        status, body = self.request("GET", "/documents/search?q=Verbrauch")
        self.assertEqual(status, 200)
        result = body["results"][0]
        self.assertEqual(
            (result["doc_date"], result["category"], result["language"]),
            ("2024-03-12", "home", "de"),
        )
        self.assertIn("ocr_quality", result)
        status, body = self.request("GET", "/unified/search?q=Verbrauch&scope=documents")
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["category"], "home")

    def test_reindex_flag_is_validated_and_accepted(self):
        self.request("POST", "/documents/index", {"max_files": 10})
        self.assertEqual(self.request("POST", "/documents/index", {"reindex": "yes"})[0], 400)
        status, body = self.request("POST", "/documents/index", {"reindex": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["unchanged"], 1)
        server.DOCUMENTS_CONTEXT_HEADER = True
        try:
            status, body = self.request("POST", "/documents/index", {"reindex": True})
            self.assertEqual((status, body["changed"]), (200, 1))
            self.assertEqual(self.request("POST", "/documents/index", {})[1]["changed"], 0)
        finally:
            server.DOCUMENTS_CONTEXT_HEADER = False

    def test_main_store_results_are_unchanged(self):
        main = SQLiteStore(":memory:", 8)
        try:
            main.upsert_file("m.md", "d", [("m1", "H", "plain needle", 1)])
            results = server.hybrid_search("needle", 3, main, None)
            self.assertNotIn("category", results[0])
        finally:
            main.close()


class ConfigFlagTests(unittest.TestCase):
    def test_context_header_defaults_off_and_keys_default(self):
        import importlib
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DOCUMENTS_CONTEXT_HEADER", None)
            os.environ.pop("DOCUMENTS_DATE_FRONTMATTER_KEYS", None)
            os.environ.pop("DOCUMENTS_ROOT", None)
            module = importlib.reload(server)
            self.assertFalse(module.DOCUMENTS_CONTEXT_HEADER)
            self.assertEqual(module.DOCUMENTS_DATE_FRONTMATTER_KEYS[0], "document_date")
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_CONTEXT_HEADER": "On",
                "DOCUMENTS_DATE_FRONTMATTER_KEYS": " Created , date ",
            },
        ):
            os.environ.pop("DOCUMENTS_ROOT", None)
            module = importlib.reload(server)
            self.assertTrue(module.DOCUMENTS_CONTEXT_HEADER)
            self.assertEqual(module.DOCUMENTS_DATE_FRONTMATTER_KEYS, ("created", "date"))
        importlib.reload(server)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
