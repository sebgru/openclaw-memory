import hashlib
import json
import math
import re
from urllib.request import Request, urlopen


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


class EmbeddingClient:
    """OpenAI-compatible embedding endpoint, with an offline deterministic fallback."""

    def __init__(self, url=None, model="default", dimensions=128, timeout=10):
        self.url, self.model, self.dimensions, self.timeout = url, model, dimensions, timeout

    def embed(self, text):
        if not self.url:
            return hash_embedding(text, self.dimensions)
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
        if len(vector) != self.dimensions:
            raise ValueError("embedding dimensions do not match configuration")
        return vector
