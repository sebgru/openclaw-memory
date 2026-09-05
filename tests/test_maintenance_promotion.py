import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from memory_store.maintenance import backup_database, verify_database
from memory_store.promotion import PromotionError, candidates, promote
from memory_store.indexer import Indexer
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


if __name__ == "__main__":
    unittest.main()
