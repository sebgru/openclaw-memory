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


if __name__ == "__main__":
    unittest.main()
