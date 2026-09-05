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
        finally:
            server.store.search = original


if __name__ == "__main__":
    unittest.main()
