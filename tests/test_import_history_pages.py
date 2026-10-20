import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from import_history_pages import import_pages


def msg(i, role="user", content=None):
    return {"id": f"m{i}", "role": role, "timestamp": f"t{i}", "content": content or f"text {i}"}


class ImportHistoryPagesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src, self.dst, self.manifest = self.tmp / "src", self.tmp / "dst", self.tmp / "m.json"
        self.src.mkdir()

    def write(self, name, data):
        (self.src / name).write_text(json.dumps(data))

    def test_merges_overlapping_pages_and_dedupes(self):
        self.write("p1.json", {"sessionKey": "agent:main:x", "messages": [msg(1), msg(2)]})
        self.write("p2.json", {"sessionKey": "agent:main:x", "messages": [msg(2), msg(3)]})
        result = import_pages(self.src, self.dst, self.manifest)
        self.assertEqual(result, {"imported": 1, "skipped": 0})
        body = next(self.dst.glob("*.md")).read_text()
        self.assertEqual(body.count("## user"), 3)
        self.assertLess(body.index("text 1"), body.index("text 3"))
        self.assertEqual(json.loads(self.manifest.read_text())[0]["messages"], 3)

    def test_block_content_and_bare_list_with_key_flag(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "tool"}]
        self.write("p.json", [msg(1, "assistant", blocks)])
        import_pages(self.src, self.dst, self.manifest, session_key="k")
        self.assertIn("hello", next(self.dst.glob("*.md")).read_text())

    def test_never_overwrites_existing(self):
        self.write("p.json", {"sessionKey": "k", "messages": [msg(1)]})
        import_pages(self.src, self.dst, self.manifest)
        target = next(self.dst.glob("*.md"))
        target.write_text("keep")
        again = import_pages(self.src, self.dst, self.manifest)
        self.assertEqual(again, {"imported": 0, "skipped": 1})
        self.assertEqual(target.read_text(), "keep")

    def test_rejects_bad_input(self):
        self.write("p.json", {"messages": [msg(1)]})
        with self.assertRaises(ValueError):
            import_pages(self.src, self.dst, self.manifest)
        self.write("p.json", {"sessionKey": "k", "messages": "nope"})
        with self.assertRaises(ValueError):
            import_pages(self.src, self.dst, self.manifest)
        with self.assertRaises(FileNotFoundError):
            import_pages(self.tmp / "missing", self.dst, self.manifest)


if __name__ == "__main__":
    unittest.main()
