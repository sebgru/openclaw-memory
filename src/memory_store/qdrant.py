import json
import uuid
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def point_id(chunk_id):
    """Return the Qdrant point ID derived from a SQLite chunk ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    """Minimal Qdrant REST adapter; no SDK or credentials are required."""

    # Qdrant rejects oversized payloads; batch to stay well under limits.
    DEFAULT_BATCH_SIZE = 256

    def __init__(self, url, collection="memory", dimensions=128, timeout=10, batch_size=None):
        self.base, self.collection, self.dimensions, self.timeout = (
            url.rstrip("/"),
            collection,
            dimensions,
            timeout,
        )
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE

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
                "id": point_id(cid),
                "vector": embed(f"{heading} {body}"),
                "payload": {"path": path, "heading": heading, "text": body, "line": line},
            }
            for cid, path, heading, body, line in records
        ]
        self._upsert_points(points)

    def upsert_precomputed(self, records):
        """Upsert records whose embedding vectors were computed by the caller."""
        if not records:
            return
        self.ensure_collection()
        points = [
            {
                "id": point_id(cid),
                "vector": vector,
                "payload": {"path": path, "heading": heading, "text": body, "line": line},
            }
            for cid, path, heading, body, line, vector in records
        ]
        self._upsert_points(points)

    def _upsert_points(self, points):
        """Send points to Qdrant in batches to avoid oversized requests."""
        endpoint = "/collections/" + quote(self.collection, safe="") + "/points"
        for offset in range(0, len(points), self.batch_size):
            batch = points[offset : offset + self.batch_size]
            self._request("PUT", endpoint, {"points": batch})

    def delete_ids(self, ids):
        if not ids:
            return
        self._request(
            "POST",
            "/collections/" + quote(self.collection, safe="") + "/points/delete",
            {"points": [point_id(item) for item in ids]},
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

    def point_count(self):
        return self.count_points(exact=True)

    def count_points(self, exact):
        result = self._request(
            "POST",
            "/collections/" + quote(self.collection, safe="") + "/points/count",
            {"exact": exact},
        )
        return int(result.get("result", {}).get("count", 0))

    def point_ids(self, ids):
        if not ids:
            return set()
        result = self._request(
            "POST",
            "/collections/" + quote(self.collection, safe="") + "/points",
            {
                "ids": [point_id(cid) for cid in ids],
                "with_payload": False,
                "with_vector": False,
            },
        )
        return {str(item["id"]) for item in result.get("result", [])}

    def diagnostics(self, store):
        chunks = [row[0] for row in store.db.execute("SELECT id FROM chunks")]
        present = self.point_ids(chunks)
        expected = {point_id(cid) for cid in chunks}
        return {
            "points": self.point_count(),
            "chunks": len(chunks),
            "missing_vectors": len(expected - present),
        }

    def bounded_diagnostics(self, store, sample_size=128):
        """Return routine status without materializing the entire chunk corpus."""
        with store._read_connection() as db:
            sample = [
                row[0]
                for row in db.execute("SELECT id FROM chunks ORDER BY id LIMIT ?", (sample_size,))
            ]
        present = self.point_ids(sample)
        expected = {point_id(chunk_id) for chunk_id in sample}
        return {
            "points": self.count_points(exact=False),
            "chunks": None,
            "missing_vectors": None,
            "sample": {
                "size": len(sample),
                "missing_vectors": len(expected - present),
                "complete_parity": False,
            },
            "parity": "sampled",
        }

    def reconcile_missing(self, store, embed, limit=1000):
        rows = store.db.execute("SELECT id, path, heading, body, line FROM chunks").fetchall()
        present = self.point_ids([row[0] for row in rows])
        missing = [row for row in rows if point_id(row[0]) not in present]
        selected = missing[:limit]
        self.upsert([(r[0], r[1], r[2], r[3], r[4]) for r in selected], embed)
        return {"reconciled": len(selected), "remaining": len(missing) - len(selected)}
