import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from unittest.mock import patch

from memory_store.qdrant import QdrantStore


class QdrantTests(unittest.TestCase):
    def test_upsert_and_search_payload(self):
        calls = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return (
                    b'{"result":[{"id":"x","score":0.9,"payload":{"path":"a.md","text":"hello"}}]}'
                )

        def fake(req, timeout):
            calls.append((req.get_method(), req.full_url))
            return Response()

        with patch("memory_store.qdrant.urlopen", fake):
            q = QdrantStore("http://qdrant:6333")
            q.upsert([("x", "a.md", "", "hello", 1)], lambda _: [0.1] * 128)
            result = q.search([0.1] * 128)
        self.assertEqual(result[0]["path"], "a.md")
        self.assertGreaterEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
