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
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_store import nimble, server

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
        nimble._reset_state()
        self.addCleanup(nimble._reset_state)
        warm = mock.patch.object(nimble, "_start_warm")
        self.warm = warm.start()
        self.addCleanup(warm.stop)
        Stub.mode, Stub.calls = "ok", 0
        self.stub.requests.clear()
        self.saved = (
            server.hybrid_search,
            server.archive_store,
            server.documents_store,
        )
        server.hybrid_search = lambda *a, **k: copy.deepcopy(ROWS)
        server.archive_store = server.documents_store = None
        for name in (
            "NIMBLE_RERANK_URL",
            "NIMBLE_RERANK_TIMEOUT",
            "NIMBLE_RERANK_MAX_CANDIDATES",
            "NIMBLE_RERANK_COOLDOWN",
        ):
            os.environ.pop(name, None)

    def tearDown(self):
        server.hybrid_search, server.archive_store, server.documents_store = self.saved
        for name in (
            "NIMBLE_RERANK_URL",
            "NIMBLE_RERANK_TIMEOUT",
            "NIMBLE_RERANK_MAX_CANDIDATES",
            "NIMBLE_RERANK_COOLDOWN",
        ):
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

    def test_exact_precedence_applies_even_when_only_one_non_exact_candidate_is_scored(
        self,
    ):
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
                nimble._reset_state()
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


class FakeResponse:
    def __init__(self, status=200, body=b"{}"):
        self.status = status
        self._body = body

    def read(self, _size):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class NimbleUnitTests(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:9"
    ENDPOINT = BASE_URL + "/v1/systemone"
    ENV = (
        "NIMBLE_RERANK_URL",
        "NIMBLE_RERANK_TIMEOUT",
        "NIMBLE_RERANK_MAX_CANDIDATES",
        "NIMBLE_RERANK_COOLDOWN",
    )

    def setUp(self):
        nimble._reset_state()
        self.addCleanup(nimble._reset_state)
        warm = mock.patch.object(nimble, "_start_warm")
        self.warm = warm.start()
        self.addCleanup(warm.stop)
        for name in self.ENV:
            os.environ.pop(name, None)
        os.environ["NIMBLE_RERANK_URL"] = self.BASE_URL

    def tearDown(self):
        for name in self.ENV:
            os.environ.pop(name, None)

    def test_invalid_numeric_settings_disable_rerank(self):
        for name in ("NIMBLE_RERANK_TIMEOUT", "NIMBLE_RERANK_MAX_CANDIDATES"):
            for bad in ("nope", ""):
                with self.subTest(name=name, value=bad):
                    os.environ[name] = bad
                    self.assertIsNone(nimble.settings())
                    before = copy.deepcopy(ROWS)
                    with mock.patch.object(
                        nimble, "_score", side_effect=AssertionError("should not score")
                    ):
                        self.assertEqual(nimble.rerank("zeta", ROWS), before)
            os.environ.pop(name, None)
        self.assertEqual(nimble.settings(), (self.ENDPOINT, 3.0, 10))

    def test_timeout_is_capped_below_caller_deadline(self):
        os.environ["NIMBLE_RERANK_TIMEOUT"] = "60"
        self.assertEqual(nimble.settings()[1], 4.0)

    def test_failure_starts_cooldown_and_single_warmup(self):
        with mock.patch.object(nimble, "_score", side_effect=OSError("down")) as score:
            self.assertEqual(nimble.rerank("zeta", ROWS), ROWS)
            self.assertEqual(nimble.rerank("zeta", ROWS), ROWS)
            self.assertEqual(score.call_count, 1)
        self.assertEqual(self.warm.call_count, 1)
        nimble._skip_until = 0.0
        with mock.patch.object(nimble, "_score", side_effect=OSError("down")):
            nimble.rerank("zeta", ROWS)
        self.assertEqual(self.warm.call_count, 1)

    def test_score_rejects_non_200_status(self):
        with (
            mock.patch.object(
                nimble.urllib.request, "urlopen", return_value=FakeResponse(status=201)
            ),
            self.assertRaises(nimble.RerankError) as raised,
        ):
            nimble._score(self.ENDPOINT, 1.0, "q", "t")
        self.assertEqual(str(raised.exception), "bad systemone status")

    def test_score_rejects_oversized_response(self):
        body = b"x" * (nimble.MAX_RESPONSE_BYTES + 1)
        with (
            mock.patch.object(
                nimble.urllib.request, "urlopen", return_value=FakeResponse(body=body)
            ),
            self.assertRaises(nimble.RerankError) as raised,
        ):
            nimble._score(self.ENDPOINT, 1.0, "q", "t")
        self.assertEqual(str(raised.exception), "systemone response too large")

    def test_all_exact_results_never_call_nimble(self):
        exact_only = [
            dict(ROWS[1], id="b1", path="b1.md"),
            dict(ROWS[1], id="b2", path="b2.md"),
        ]
        before = copy.deepcopy(exact_only)
        with mock.patch.object(nimble, "_score", side_effect=AssertionError("should not score")):
            self.assertEqual(nimble.rerank("zeta", exact_only), before)

    def test_expired_budget_returns_baseline_without_scoring(self):
        candidates = [dict(ROWS[0]), dict(ROWS[2])]
        before = copy.deepcopy(candidates)
        with mock.patch.object(nimble, "settings", return_value=(self.ENDPOINT, 0.0, 10)):
            with mock.patch.object(
                nimble, "_score", side_effect=AssertionError("should not score")
            ):
                self.assertEqual(nimble.rerank("zeta", candidates), before)


if __name__ == "__main__":
    unittest.main()
