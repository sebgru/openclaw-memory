"""Optional, default-off Nimble reranking. Local Ollama; all-or-nothing."""

import ipaddress
import json
import math
import os
import time
import urllib.request
from urllib.parse import urlparse

LOCAL_HOSTS = {"localhost", "ollama-coder"}
MAX_RESPONSE_BYTES = 1_000_000
MAX_TEXT_CHARS = 8_000
MODEL = "nimble:9b"


class RerankError(Exception):
    pass


def settings():
    """Return (endpoint, deadline, candidate cap) or None when disabled/invalid."""
    endpoint = os.getenv("NIMBLE_RERANK_URL", "").strip()
    if not endpoint:
        return None
    parsed = urlparse(endpoint)
    host = (parsed.hostname or "").lower()
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host in LOCAL_HOSTS
    if (
        parsed.scheme != "http"
        or not local
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return None
    try:
        deadline = min(8.0, max(0.05, float(os.getenv("NIMBLE_RERANK_TIMEOUT", "8.0"))))
        cap = min(10, max(1, int(os.getenv("NIMBLE_RERANK_MAX_CANDIDATES", "10"))))
    except ValueError:
        return None
    return endpoint.rstrip("/") + "/v1/systemone", deadline, cap


def _probability(payload):
    answer = payload.get("answers", {}).get("relevance") if isinstance(payload, dict) else None
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    relevant = probabilities.get("relevant") if isinstance(probabilities, dict) else None
    not_relevant = probabilities.get("not_relevant") if isinstance(probabilities, dict) else None
    if (
        not isinstance(answer, dict)
        or answer.get("type") != "choice"
        or answer.get("choice") not in {"relevant", "not_relevant"}
        or isinstance(relevant, bool)
        or isinstance(not_relevant, bool)
        or not isinstance(relevant, (int, float))
        or not isinstance(not_relevant, (int, float))
        or not math.isfinite(relevant)
        or not math.isfinite(not_relevant)
        or not 0 <= relevant <= 1
        or not 0 <= not_relevant <= 1
    ):
        raise RerankError("malformed systemone choice result")
    return float(relevant)


def _score(endpoint, timeout, query, text):
    body = json.dumps(
        {
            "model": MODEL,
            "state": f"Query:\n{query}\n\nCandidate:\n{text[:MAX_TEXT_CHARS]}",
            "questions": {
                "relevance": {
                    "type": "choice",
                    "instructions": "Judge whether this candidate is useful evidence for answering the query. Judge relevance only, not truth.",
                    "criteria": {
                        "relevant": "Directly answers the query or contains useful, specific evidence for it.",
                        "not_relevant": "Does not materially help answer the query.",
                    },
                }
            },
            "keep_alive": "60s",
        }
    ).encode()
    request = urllib.request.Request(
        endpoint, body, {"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        if response.status != 200:
            raise RerankError("bad systemone status")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RerankError("systemone response too large")
    try:
        return _probability(json.loads(raw))
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise RerankError("malformed systemone response") from exc


def rerank(query, results):
    """Reorder bounded non-exact candidates; any failure returns baseline intact."""
    config = settings()
    if config is None or len(results) < 2:
        return results
    endpoint, deadline, cap = config
    baseline = list(results)
    exact = [item for item in baseline if item.get("exact_match")]
    selected = [item for item in baseline if not item.get("exact_match")][:cap]
    selected_ids = {id(item) for item in selected}
    remaining_non_exact = [
        item for item in baseline if not item.get("exact_match") and id(item) not in selected_ids
    ]
    if not selected:
        return baseline
    if len(selected) < 2:
        return exact + selected + remaining_non_exact
    started = time.monotonic()
    scores = {}
    try:
        for index, item in enumerate(selected):
            remaining = deadline - (time.monotonic() - started)
            if remaining <= 0:
                raise RerankError("total scoring deadline exceeded")
            scores[index] = _score(
                endpoint,
                remaining,
                query,
                str(item.get("text", "")),
            )
    except Exception:  # noqa: BLE001 - no partial ordering may escape
        return baseline

    selected_sorted = [
        item
        for _, item in sorted(enumerate(selected), key=lambda pair: (-scores[pair[0]], pair[0]))
    ]
    return exact + selected_sorted + remaining_non_exact
