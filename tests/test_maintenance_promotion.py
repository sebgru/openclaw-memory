import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from memory_store.chunker import chunk_markdown
from memory_store.indexer import Indexer
from memory_store.maintenance import backup_database, verify_database
from memory_store.promotion import PromotionError, audit_record, candidates, promote
from memory_store.sqlite_store import SQLiteStore


class MaintenanceTests(unittest.TestCase):
    def test_archive_index_is_incremental_and_removes_deleted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "session.md"
            source.write_text("# Session\nold evidence")
            store = SQLiteStore(":memory:")
            self.assertEqual(Indexer(root, store).scan().as_dict()["added"], 1)
            source.write_text("# Session\nnew evidence")
            self.assertEqual(Indexer(root, store).scan().as_dict()["changed"], 1)
            source.unlink()
            self.assertEqual(Indexer(root, store).scan().as_dict()["removed"], 1)
            self.assertFalse(store.search("new"))
            store.close()

    def test_backup_and_restore_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.db"
            target = root / "backup.db"
            store = SQLiteStore(source)
            store.upsert_file("a.md", "digest", [("id", "Heading", "body", 1)])
            store.close()
            result = backup_database(source, target)
            self.assertEqual(result["verification"]["status"], "ok")
            self.assertEqual(result["restore_verification"]["integrity"], "ok")
            self.assertFalse((root / "backup.db.restore-check").exists())

    def test_integrity_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.db"
            path.write_bytes(b"not sqlite")
            self.assertEqual(verify_database(path)["status"], "error")

    def test_backup_without_restore_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.db"
            target = Path(directory) / "backup.db"
            SQLiteStore(source).close()
            result = backup_database(source, target, verify_restore=False)
            self.assertNotIn("restore_verification", result)
            self.assertEqual(result["verification"]["status"], "ok")


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.candidate = self.root / "memory/review-candidates/fact.md"
        self.candidate.parent.mkdir(parents=True)
        self.candidate.write_text("# Fact\nKeep this")

    def tearDown(self):
        self.tmp.cleanup()

    def test_candidates_are_listed_without_promotion(self):
        self.assertEqual(candidates(self.root)[0]["path"], "memory/review-candidates/fact.md")
        self.assertFalse((self.root / "memory/knowledge/fact.md").exists())

    def test_promotion_requires_human_approval_and_retains_source(self):
        with self.assertRaises(PromotionError):
            promote(self.root, "memory/review-candidates/fact.md", "memory/knowledge/fact.md", "")
        result = promote(
            self.root, "memory/review-candidates/fact.md", "memory/knowledge/fact.md", "Sebastian"
        )
        self.assertEqual(result["approved_by"], "Sebastian")
        self.assertIn("promoted from", (self.root / "memory/knowledge/fact.md").read_text())
        self.assertTrue(self.candidate.exists())

    def test_promotion_rejects_unsafe_paths(self):
        with self.assertRaises(PromotionError):
            promote(self.root, "memory/review-candidates/fact.md", "MEMORY.md", "human")

    def test_candidates_returns_empty_for_missing_queue(self):
        self.assertEqual(candidates(self.root / "nonexistent"), [])

    def test_candidates_marks_empty_files_ineligible(self):
        empty = self.candidate.parent / "empty.md"
        empty.write_text("\n")
        result = {item["path"]: item for item in candidates(self.root)}
        self.assertEqual(result["memory/review-candidates/empty.md"]["eligible"], "false")
        self.assertEqual(result["memory/review-candidates/empty.md"]["reason"], "candidate is empty")

    def test_promotion_does_not_overwrite_reviewed_destination(self):
        target = self.root / "memory/knowledge/fact.md"
        target.parent.mkdir(parents=True)
        target.write_text("existing")
        with self.assertRaises(PromotionError):
            promote(self.root, "memory/review-candidates/fact.md", "memory/knowledge/fact.md", "human")

    def test_promotion_rejects_missing_candidate(self):
        with self.assertRaises(PromotionError):
            promote(
                self.root, "memory/review-candidates/missing.md", "memory/knowledge/f.md", "human"
            )

    def test_promotion_rejects_non_markdown_destination(self):
        with self.assertRaises(PromotionError):
            promote(
                self.root, "memory/review-candidates/fact.md", "memory/knowledge/f.txt", "human"
            )

    def test_audit_record_is_stable_json(self):
        record = audit_record({"candidate": "c.md", "approved_by": "human"})
        self.assertEqual(record, '{"approved_by": "human", "candidate": "c.md"}')

    def test_chunker_splits_oversized_paragraph_and_rejects_small_max(self):
        with self.assertRaises(ValueError):
            chunk_markdown("x", max_chars=50)
        huge = "# H\n\n" + ("word " * 300).strip()
        chunks = chunk_markdown(huge, max_chars=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.text) <= 100 for c in chunks))

    def test_chunker_handles_oversized_paragraph_mid_section(self):
        first = "word " * 60
        second = "tail " * 60
        chunks = chunk_markdown(f"# H\n\n{first.strip()}\n\n{second.strip()}", max_chars=150)
        self.assertTrue(any("word" in c.text for c in chunks[:1]))
        self.assertTrue(all("tail" in c.text for c in chunks[-2:]))
        self.assertTrue(all(len(c.text) <= 150 for c in chunks))


if __name__ == "__main__":
    unittest.main()
