import sys
import sqlite3
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from unittest.mock import patch

from memory_store.qdrant import QdrantStore


class QdrantTests(unittest.TestCase):
    def test_bounded_diagnostics_checks_only_a_sample_and_uses_approximate_count(self):
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE chunks (id TEXT)")
        db.executemany("INSERT INTO chunks VALUES (?)", [(str(i),) for i in range(1000)])

        class Store:
            def _read_connection(self):
                from contextlib import nullcontext

                return nullcontext(db)

        q = QdrantStore("http://qdrant:6333")
        count_requests = []
        q._request = lambda method, path, payload=None: (
            count_requests.append(payload) or {"result": {"count": 1000}}
        )
        q.point_ids = lambda ids: set()
        result = q.bounded_diagnostics(Store(), sample_size=8)
        self.assertIsNone(result["chunks"])
        self.assertEqual(result["sample"]["size"], 8)
        self.assertEqual(result["sample"]["missing_vectors"], 8)
        self.assertIsNone(result["missing_vectors"])
        self.assertEqual(count_requests, [{"exact": False}])

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
