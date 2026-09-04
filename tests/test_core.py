import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from memory_store.chunker import chunk_markdown
from memory_store.indexer import Indexer
from memory_store.sqlite_store import SQLiteStore


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = SQLiteStore()

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_chunking_preserves_headings(self):
        chunks = chunk_markdown("# One\nalpha\n\n## Two\nbeta")
        self.assertEqual([c.heading for c in chunks], ["One", "Two"])

    def test_incremental_and_removal(self):
        file = self.root / "a.md"
        file.write_text("# Topic\nhello world")
        self.assertEqual(
            Indexer(self.root, self.store).scan().as_dict(),
            {"added": 1, "changed": 0, "removed": 0, "unchanged": 0},
        )
        self.assertEqual(Indexer(self.root, self.store).scan().unchanged, 1)
        self.assertTrue(self.store.search("hello"))
        file.unlink()
        self.assertEqual(Indexer(self.root, self.store).scan().removed, 1)
        self.assertFalse(self.store.search("hello"))

    def test_missing_root(self):
        with self.assertRaises(FileNotFoundError):
            Indexer(self.root / "missing", self.store).scan()

    def test_include_and_exclude_patterns(self):
        (self.root / "MEMORY.md").write_text("# Durable\nkeep this")
        (self.root / "OTHER.md").write_text("# Other\nignore this")
        (self.root / "memory" / "review-candidates").mkdir(parents=True)
        (self.root / "memory" / "daily.md").write_text("# Daily\nremember this")
        (self.root / "memory" / "review-candidates" / "draft.md").write_text(
            "# Draft\nnot authoritative"
        )
        stats = Indexer(
            self.root,
            self.store,
            include_patterns=("MEMORY.md", "memory/**/*.md"),
            exclude_patterns=("memory/review-candidates/**",),
        ).scan()
        self.assertEqual(stats.added, 2)
        self.assertTrue(self.store.search("keep"))
        self.assertTrue(self.store.search("remember"))
        self.assertFalse(self.store.search("authoritative"))
        self.assertIsNone(self.store.file_digest("OTHER.md"))


if __name__ == "__main__":
    unittest.main()
