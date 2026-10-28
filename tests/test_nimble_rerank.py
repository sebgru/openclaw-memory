import copy
import json
import os
import sys
import time
import unittest
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_store import server  # noqa: E402

ROWS = [
    {
        "id": "a",
        "text": "alpha",
        "path": "a.md",
        "heading": "H1",
        "line": 1,
        "score": 0.9,
        "lexical_evidence": 0,
        "semantic_similarity": 0.3,
        "exact_match": False,
    },
    {
        "id": "b",
        "text": "bravo",
        "path": "b.md",
        "heading": "H2",
        "line": 2,
        "score": 0.8,
        "lexical_evidence": 0,
        "semantic_similarity": 0.3,
        "exact_match": True,
    },
    {
        "id": "c",
        "text": "charlie",
        "path": "c.md",
        "heading": "H3",
        "line": 3,
        "score": 0.7,
        "lexical_evidence": 0,
        "semantic_similarity": 0.3,
        "exact_match": False,
    },
    {
        "id": "d",
        "text": "delta",
        "path": "d.md",
        "heading": "H4",
        "line": 4,
        "score": 0.6,
        "lexical_evidence": 0,
        "semantic_similarity": 0.3,
        "exact_match": False,
    },
]


class Stub(BaseHTTPRequestHandler):
    mode = "ok"
    calls = 0

    def do_POST(self):
        cls = type(self)
        cls.calls += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        if cls.mode == "timeout":
            time.sleep(0.6)
        score = next(
            (
                p
                for text, p in (("alpha", 0.1), ("charlie", 0.9), ("delta", 0.4))
                if text in body.get("state", "")
            ),
            0.2,
        )
        answer = {
            "answers": {
                "relevance": {
                    "type": "choice",
                    "choice": "relevant",
                    "probabilities": {"relevant": score, "not_relevant": 1 - score},
                }
            }
        }
        code, payload = 200, answer
        if cls.mode == "http500":
            code = 500
        elif cls.mode == "malformed":
            payload = {"nope": 1}
        elif cls.mode == "late_bad" and cls.calls >= 2:
            payload = {
                "answers": {
                    "relevance": {
                        "type": "choice",
                        "choice": "relevant",
                        "probabilities": {"relevant": "bad"},
                    }
                }
            }
        data = b"not json" if cls.mode == "garbage" else json.dumps(payload).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass

    def log_message(self, *args):
        pass


class NimbleRerankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        cls.stub.requests = []
        Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        Thread(target=cls.http.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.stub.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.stub.shutdown()

    def setUp(self):
        Stub.mode, Stub.calls = "ok", 0
        self.stub.requests.clear()
        self.saved = (server.hybrid_search, server.archive_store, server.documents_store)
        server.hybrid_search = lambda *a, **k: copy.deepcopy(ROWS)
        server.archive_store = server.documents_store = None
        for name in ("NIMBLE_RERANK_URL", "NIMBLE_RERANK_TIMEOUT", "NIMBLE_RERANK_MAX_CANDIDATES"):
            os.environ.pop(name, None)

    def tearDown(self):
        server.hybrid_search, server.archive_store, server.documents_store = self.saved
        for name in ("NIMBLE_RERANK_URL", "NIMBLE_RERANK_TIMEOUT", "NIMBLE_RERANK_MAX_CANDIDATES"):
            os.environ.pop(name, None)

    def search(self):
        c = HTTPConnection(*self.http.server_address)
        c.request("GET", "/unified/search?scope=main&q=zeta")
        r = c.getresponse()
        return r.status, json.loads(r.read())

    def enable(self, timeout="8"):
        os.environ["NIMBLE_RERANK_URL"] = self.url
        os.environ["NIMBLE_RERANK_TIMEOUT"] = timeout

    def ids(self, payload):
        return [r["path"][0] for r in payload["results"]]

    def test_default_off_makes_no_calls_and_matches_baseline(self):
        status, baseline = self.search()
        self.assertEqual(status, 200)
        self.assertEqual(Stub.calls, 0)
        self.assertEqual(self.ids(baseline), ["a", "b", "c", "d"])
        os.environ["NIMBLE_RERANK_URL"] = ""
        self.assertEqual(self.search()[1], baseline)
        self.assertEqual(Stub.calls, 0)

    def test_non_local_endpoint_is_ignored(self):
        _, baseline = self.search()
        for url in ("http://example.com/rerank", "https://127.0.0.1/rerank"):
            os.environ["NIMBLE_RERANK_URL"] = url
            self.assertEqual(self.search()[1], baseline)
        self.assertEqual(Stub.calls, 0)

    def test_success_preserves_exact_precedence_and_provenance(self):
        _, baseline = self.search()
        self.enable()
        status, payload = self.search()
        self.assertEqual(status, 200)
        self.assertEqual(Stub.calls, 3)  # Exact candidate "b" is never sent to Nimble.
        self.assertEqual(self.ids(payload), ["b", "c", "d", "a"])
        self.assertEqual([r["model"] for r in self.stub.requests], ["nimble:9b"] * 3)
        self.assertTrue(
            all(request["state"].startswith("Query:\n") for request in self.stub.requests)
        )
        self.assertTrue(
            all(
                request["questions"]["relevance"]["type"] == "choice"
                for request in self.stub.requests
            )
        )
        by_id = {r["path"][0]: r for r in baseline["results"]}
        for item in payload["results"]:
            self.assertEqual(item, by_id[item["path"][0]])
        self.assertEqual(payload["coverage"], baseline["coverage"])
        self.assertEqual(payload.get("low_confidence"), baseline.get("low_confidence"))

    def test_exact_precedence_applies_even_when_only_one_non_exact_candidate_is_scored(self):
        server.hybrid_search = lambda *a, **k: copy.deepcopy(ROWS[:2])
        self.enable()
        status, payload = self.search()
        self.assertEqual(status, 200)
        self.assertEqual(self.ids(payload), ["b", "a"])
        self.assertEqual(Stub.calls, 0)

    def test_candidate_cap_and_unscored_tail(self):
        os.environ["NIMBLE_RERANK_MAX_CANDIDATES"] = "2"
        self.enable()
        status, payload = self.search()
        self.assertEqual(status, 200)
        self.assertEqual(Stub.calls, 2)
        self.assertEqual(self.ids(payload), ["b", "c", "a", "d"])

    def test_failures_fall_back_to_exact_baseline(self):
        _, baseline = self.search()
        self.enable(timeout="0.05")
        for mode in ("timeout", "http500", "malformed", "late_bad", "garbage"):
            with self.subTest(mode=mode):
                Stub.mode, Stub.calls = mode, 0
                started = time.monotonic()
                status, payload = self.search()
                self.assertLess(time.monotonic() - started, 0.55)
                self.assertEqual(status, 200)
                self.assertEqual(payload, baseline)

    def test_transport_error_falls_back(self):
        _, baseline = self.search()
        self.stub.shutdown()
        self.stub.server_close()
        try:
            self.enable()
            self.assertEqual(self.search()[1], baseline)
        finally:
            type(self).stub = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
            type(self).stub.requests = []
            Thread(target=type(self).stub.serve_forever, daemon=True).start()
            type(self).url = f"http://127.0.0.1:{type(self).stub.server_address[1]}"


if __name__ == "__main__":
    unittest.main()
