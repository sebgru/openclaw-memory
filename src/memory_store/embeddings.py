import hashlib
import json
import logging
import math
import re
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("memory_store.embeddings")


def hash_embedding(text: str, dimensions: int = 128) -> list[float]:
    """Deterministic, dependency-free baseline embedding."""
    if dimensions < 8:
        raise ValueError("dimensions must be at least 8")
    vector = [0.0] * dimensions
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def validate_vector(vector: list[float], dimensions: int) -> None:
    """Raise ValueError if a vector is malformed (wrong size, NaN, Inf)."""
    if len(vector) != dimensions:
        raise ValueError(f"embedding dimensions {len(vector)} do not match configured {dimensions}")
    for i, v in enumerate(vector):
        if not math.isfinite(v):
            raise ValueError(f"embedding contains non-finite value at index {i}: {v!r}")


class EmbeddingClient:
    """OpenAI-compatible embedding endpoint, with an offline deterministic fallback."""

    MAX_RETRIES = 3
    RETRY_BACKOFF = 0.5  # seconds; doubles each attempt

    def __init__(self, url=None, model="default", dimensions=128, timeout=10):
        self.url, self.model, self.dimensions, self.timeout = url, model, dimensions, timeout

    def embed(self, text):
        if not self.url:
            return hash_embedding(text, self.dimensions)
        for attempt in range(self.MAX_RETRIES):
            try:
                vector = self._fetch(text)
                validate_vector(vector, self.dimensions)
                return vector
            except (URLError, OSError, ValueError, KeyError) as exc:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_BACKOFF * (2**attempt)
                    logger.warning(
                        "embedding attempt %d/%d failed for %r: %s; retrying in %.1fs",
                        attempt + 1,
                        self.MAX_RETRIES,
                        text[:60],
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        # All remote attempts failed; fall back to deterministic hash embedding.
        logger.warning(
            "embedding service unavailable after %d attempts; using hash fallback for %r",
            self.MAX_RETRIES,
            text[:60],
        )
        if self.dimensions < 8:
            # hash_embedding requires dimensions >= 8; use simple deterministic fallback.
            return self._simple_fallback(text)
        return hash_embedding(text, self.dimensions)

    def _simple_fallback(self, text):
        """Deterministic fallback for dimensions < 8 (hash_embedding minimum)."""
        digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
        vector = [0.0] * self.dimensions
        for i in range(0, min(16, self.dimensions * 2), 2):
            vector[i // 2 % self.dimensions] += (
                int.from_bytes(digest[i : i + 2], "big", signed=True) / 32768.0
            )
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def _fetch(self, text):
        request = Request(
            self.url,
            data=json.dumps({"model": self.model, "input": [text]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        # Accept both OpenAI-compatible `{data: [{embedding: [...] }]}` and
        # Ollama `/api/embed` `{embeddings: [[...]]}` responses.
        vector = payload.get("data", [{}])[0].get("embedding") if payload.get("data") else None
        if vector is None:
            vector = payload["embeddings"][0]
        return list(vector)
