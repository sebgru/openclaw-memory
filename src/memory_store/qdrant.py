import json
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


class QdrantStore:
    """Minimal Qdrant REST adapter; no SDK or credentials are required."""

    def __init__(self, url, collection="memory", dimensions=128, timeout=10):
        self.base, self.collection, self.dimensions, self.timeout = (
            url.rstrip("/"),
            collection,
            dimensions,
            timeout,
        )

    def _request(self, method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(
            self.base + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        with urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    def ensure_collection(self):
        path = "/collections/" + quote(self.collection, safe="")
        try:
            info = self._request("GET", path)
        except HTTPError as exc:
            if exc.code != 404:
                raise
            self._request("PUT", path, {"vectors": {"size": self.dimensions, "distance": "Cosine"}})
            return
        result = info.get("result", {})
        if not isinstance(result, dict):
            return
        vectors = result.get("config", {}).get("params", {}).get("vectors", {})
        size = vectors.get("size")
        if size is not None and size != self.dimensions:
            raise ValueError(
                f"Qdrant collection dimension {size} does not match configured {self.dimensions}"
            )

    def upsert(self, records, embed):
        if not records:
            return
        self.ensure_collection()
        points = [
            {
                "id": cid,
                "vector": embed(f"{heading} {body}"),
                "payload": {"path": path, "heading": heading, "text": body, "line": line},
            }
            for cid, path, heading, body, line in records
        ]
        self._request(
            "PUT", "/collections/" + quote(self.collection, safe="") + "/points", {"points": points}
        )

    def delete_file(self, path):
        self._request(
            "POST",
            "/collections/" + quote(self.collection, safe="") + "/points/delete",
            {"filter": {"must": [{"key": "path", "match": {"value": path}}]}},
        )

    def search(self, vector, limit=20):
        result = self._request(
            "POST",
            "/collections/" + quote(self.collection, safe="") + "/points/search",
            {"vector": vector, "limit": limit, "with_payload": True},
        )
        return [
            {"id": str(x["id"]), **x.get("payload", {}), "score": x.get("score", 0.0)}
            for x in result.get("result", [])
        ]
