import json
from pathlib import Path

import pytest

from memory_store.mail_index import build
from memory_store.sqlite_store import SQLiteStore


def snapshot(root: Path, files: dict[str, str]):
    current = root / "current"
    mail = current / "mail"
    mail.mkdir(parents=True, exist_ok=True)
    (current / "state.json").write_text(json.dumps({"version": 1}))
    for p in mail.glob("*.md"):
        p.unlink()
    for name, text in files.items():
        (mail / name).write_text(text)


def test_incremental_reconciliation(tmp_path, monkeypatch):
    monkeypatch.delenv("EMBEDDING_URL", raising=False)
    root, db = tmp_path / "export", tmp_path / "mail.db"
    snapshot(root, {"one.md": "# Invoice\nFrom: Example <person@example.invalid>\nRoof repair"})
    assert build(str(root), str(db))["added"] == 1
    assert build(str(root), str(db))["unchanged"] == 1
    snapshot(root, {"one.md": "# Invoice\nRoof replaced", "two.md": "# Warranty\nSeven years"})
    stats = build(str(root), str(db))
    assert stats["added"] == stats["changed"] == 1
    snapshot(root, {})
    assert build(str(root), str(db))["removed"] == 2
    store = SQLiteStore(str(db))
    assert store.db.execute("SELECT count(*) FROM files").fetchone()[0] == 0
    store.db.close()


def test_missing_and_collision(tmp_path, monkeypatch):
    from memory_store.mail_index import _build

    root = tmp_path / "export"
    with pytest.raises(ValueError, match="completed"):
        build(str(root), str(tmp_path / "mail.db"))
    with pytest.raises(ValueError, match="completed"):
        _build(str(root), str(tmp_path / "mail.db"))
    snapshot(root, {})
    with pytest.raises(ValueError, match="distinct"):
        build(str(root), str(tmp_path / "memory.db"))
    db = tmp_path / "other.db"
    monkeypatch.setenv("SQLITE_PATH", str(db))
    with pytest.raises(ValueError, match="collides"):
        build(str(root), str(db))


def test_private_mail_scope_and_authentication(tmp_path, monkeypatch):
    from http.client import HTTPConnection
    from http.server import ThreadingHTTPServer
    from threading import Thread

    from memory_store import server

    mail = SQLiteStore(str(tmp_path / "private.sqlite"))
    monkeypatch.setattr(server, "mail_store", mail)
    monkeypatch.setattr(server, "MAIL_TOKEN", "invented-test-capability")
    seen = []

    def search(query, limit, selected_store=None, selected_vector_store=None):
        seen.append(selected_store)
        return [{"id": "invented", "path": "invented.md", "text": "Roof warranty", "line": 1}]

    monkeypatch.setattr(server, "hybrid_search", search)
    http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()

    def request(scope, token=""):
        connection = HTTPConnection(*http.server_address)
        connection.request(
            "GET", f"/unified/search?q=roof&scope={scope}", headers={"X-Mail-Access": token}
        )
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    try:
        assert request("mail")[0] == 403
        assert request("mail", "wrong")[0] == 403
        assert request("mail", "é")[0] == 403
        assert seen == []
        status, payload = request("mail", "invented-test-capability")
        assert status == 200
        assert payload["results"][0]["source"] == "mail"
        assert payload["coverage"]["mail"] == "searched"
        seen.clear()
        assert request("all")[0] == 200
        assert mail not in seen
        monkeypatch.setattr(server, "MAIL_TOKEN", "")
        assert request("mail", "invented-test-capability")[0] == 403
    finally:
        http.shutdown()
        http.server_close()
        thread.join()
        mail.db.close()


def test_index_shares_export_lock_and_rejects_relative_database(tmp_path):
    root = tmp_path / "export"
    snapshot(root, {})
    (root / ".sync.lock").write_text("")
    with pytest.raises(FileExistsError):
        build(str(root), str(tmp_path / "mail.db"))
    (root / ".sync.lock").unlink()
    with pytest.raises(ValueError, match="absolute"):
        build(str(root), "relative-mail.sqlite")
    assert not (root / ".sync.lock").exists()


def test_structured_provenance_and_optional_vectors(tmp_path, monkeypatch, capsys):
    from memory_store import mail_index

    root, db = tmp_path / "export", tmp_path / "mail.sqlite"
    snapshot(root, {"message.md": "# Warranty\nSeven-year warranty"})
    state = {
        "aliases": {"invented-id": "message.md"},
        "records": {
            "invented-id": {
                "subject": "Warranty",
                "receivedDateTime": "2026-01-01",
                "from": {"emailAddress": {"address": "person@example.invalid"}},
            }
        },
    }
    (root / "current/state.json").write_text(json.dumps(state))
    calls = []

    class Vectors:
        def __init__(self, url, collection, dimensions):
            calls.append((url, collection, dimensions))

        def ensure_collection(self):
            pass

        def delete_file(self, path):
            calls.append("delete")

        def upsert(self, records, embed):
            assert records
            calls.append("upsert")

    monkeypatch.setattr(mail_index, "MailQdrantStore", Vectors)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mail_index",
            "--export-root",
            str(root),
            "--database",
            str(db),
            "--qdrant-url",
            "http://example.invalid",
            "--collection",
            "mail-private-fixture",
        ],
    )
    mail_index.main()
    assert json.loads(capsys.readouterr().out)["added"] == 1
    assert calls == [("http://example.invalid", "mail-private-fixture", 128), "delete", "upsert"]
    store = SQLiteStore(str(db))
    assert tuple(
        store.db.execute("SELECT subject, sender, date FROM mail_metadata").fetchone()
    ) == ("Warranty", "person@example.invalid", "2026-01-01")
    store.db.close()
    assert db.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="namespace"):
        build(str(root), str(db), collection="documents")


def test_mail_lane_failure_provenance_and_message_collapse(tmp_path, monkeypatch):
    from memory_store import server

    mail = SQLiteStore(str(tmp_path / "mail.sqlite"))
    monkeypatch.setattr(server, "mail_store", mail)
    monkeypatch.setattr(server, "mail_vector_store", None)
    try:
        with mail.db:
            mail.db.execute(
                "CREATE TABLE mail_metadata(path TEXT, subject TEXT, sender TEXT, date TEXT)"
            )
            mail.db.execute(
                "INSERT INTO mail_metadata VALUES (?,?,?,?)",
                ("message.md", "Warranty", "person@example.invalid", "2026-01-01"),
            )
        monkeypatch.setattr(
            server,
            "hybrid_search",
            lambda *a: [
                {"id": "one", "path": "message.md", "text": "fact one", "line": 1},
                {"id": "two", "path": "message.md", "text": "fact two", "line": 2},
                {"id": "three", "path": "missing.md", "text": "fact three", "line": 1},
            ],
        )
        result, warnings, coverage, degraded = server.unified_search_with_coverage(
            "warranty", 3, "mail"
        )
        assert len(result) == 2
        assert result[0]["mail"]["date"] == "2026-01-01"
        assert "mail" not in result[1]
        assert warnings and degraded == {"mail": "lexical_fallback"}
        assert coverage["mail"] == "searched"
        with pytest.raises(ValueError):
            server.unified_search_with_coverage("x", 1, "mail", "invalid")
        monkeypatch.setattr(
            server, "hybrid_search", lambda *a: (_ for _ in ()).throw(RuntimeError("down"))
        )
        results, warnings, coverage, degraded = server.unified_search_with_coverage("x", 1, "mail")
        assert results == [] and coverage["mail"] == "unavailable"
        monkeypatch.setattr(server, "mail_store", None)
        with pytest.raises(LookupError):
            server.unified_search_with_coverage("x", 1, "mail")
    finally:
        mail.db.close()


def test_nullable_graph_sender(tmp_path):
    root, db = tmp_path / "export", tmp_path / "mail.sqlite"
    snapshot(root, {"message.md": "# Contract\nTerms"})
    (root / "current/state.json").write_text(
        json.dumps(
            {
                "aliases": {"invented": "message.md"},
                "records": {"invented": {"subject": "Contract", "from": None}},
            }
        )
    )
    assert build(str(root), str(db))["added"] == 1
    store = SQLiteStore(str(db))
    assert store.db.execute("SELECT sender FROM mail_metadata").fetchone()[0] == ""
    store.db.close()


def test_mail_honors_existing_fusion_configuration(tmp_path, monkeypatch):
    from memory_store import server

    mail = SQLiteStore(str(tmp_path / "mail.sqlite"))
    monkeypatch.setattr(server, "mail_store", mail)
    monkeypatch.setattr(server._search_state, "rank_fusion", False, raising=False)
    seen = []
    monkeypatch.setattr(
        server, "hybrid_search", lambda *a: seen.append(server._search_state.rank_fusion) or []
    )
    try:
        monkeypatch.setenv("UNIFIED_FUSION_MODE", "rrf")
        server.unified_search_with_coverage("query", 3, "mail")
        assert seen == [True]
        assert server._search_state.rank_fusion is False
        monkeypatch.setenv("UNIFIED_FUSION_MODE", "invalid")
        with pytest.raises(ValueError, match="UNIFIED_FUSION_MODE"):
            server.unified_search_with_coverage("query", 3, "mail")
    finally:
        mail.db.close()


class FaultVectors:
    """Persistent fake remote service, including partially applied operations."""

    def __init__(self):
        self.points = {}
        self.fail = None

    def ensure_collection(self):
        pass

    def delete_file(self, path):
        if self.fail == "before_delete":
            self.fail = None
            raise RuntimeError("injected delete failure")
        self.points = {key: row for key, row in self.points.items() if row["path"] != path}
        if self.fail == "after_delete":
            self.fail = None
            raise RuntimeError("injected delete acknowledgement failure")

    def upsert(self, records, embed):
        from memory_store.qdrant import point_id

        if self.fail == "before_upsert":
            self.fail = None
            raise RuntimeError("injected upsert failure")
        for cid, path, heading, body, line in records:
            key = point_id(cid)
            self.points[key] = {
                "id": key,
                "path": path,
                "heading": heading,
                "text": body,
                "line": line,
                "score": 0.95,
            }
            if self.fail == "partial_upsert":
                self.fail = None
                raise RuntimeError("injected partial batch failure")
        if self.fail == "after_upsert":
            self.fail = None
            raise RuntimeError("injected upsert acknowledgement failure")

    def search(self, vector, limit):
        return list(self.points.values())[:limit]


def queue_paths(database):
    import sqlite3

    with sqlite3.connect(database) as db:
        return [row[0] for row in db.execute("SELECT path FROM mail_vector_pending ORDER BY path")]


def install_vectors(monkeypatch):
    from memory_store import mail_index

    vectors = FaultVectors()
    monkeypatch.delenv("EMBEDDING_URL", raising=False)
    monkeypatch.setattr(mail_index, "MailQdrantStore", lambda *a: vectors)
    return vectors


def assert_vector_parity(database, vectors):
    from memory_store.qdrant import point_id

    store = SQLiteStore(str(database))
    try:
        expected = {
            point_id(row["id"]): (row["path"], row["body"])
            for row in store.db.execute("SELECT * FROM chunks")
        }
        assert expected == {key: (row["path"], row["text"]) for key, row in vectors.points.items()}
    finally:
        store.db.close()


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize(
    "failure", ["before_upsert", "partial_upsert", "after_upsert", "before_delete", "after_delete"]
)
def test_vector_failures_are_durable_and_converge(tmp_path, monkeypatch, changed, failure):
    root, db = tmp_path / "export", tmp_path / "mail.sqlite"
    vectors = install_vectors(monkeypatch)
    if changed:
        snapshot(root, {"one.md": "# Old terms\nObsolete warranty"})
        build(str(root), str(db), qdrant_url="http://example.invalid")
    snapshot(root, {"one.md": "# Current terms\n" + ("Current fixture facts. " * 160)})
    vectors.fail = failure
    with pytest.raises(RuntimeError, match="injected"):
        build(str(root), str(db), qdrant_url="http://example.invalid")
    assert queue_paths(db) == ["one.md"]
    # A new build reopens SQLite: digest equality must not skip pending vectors.
    assert build(str(root), str(db), qdrant_url="http://example.invalid")["unchanged"] == 1
    assert queue_paths(db) == []
    assert len(vectors.points) > 1  # partial batch probe is discriminating
    assert_vector_parity(db, vectors)
    assert build(str(root), str(db), qdrant_url="http://example.invalid")["unchanged"] == 1
    assert_vector_parity(db, vectors)


@pytest.mark.parametrize("failure", ["before_delete", "after_delete"])
@pytest.mark.parametrize("readd", [False, True])
def test_deletion_tombstones_retry_and_stale_search_is_rejected(
    tmp_path, monkeypatch, failure, readd
):
    from memory_store import server

    root, db = tmp_path / "export", tmp_path / "mail.sqlite"
    vectors = install_vectors(monkeypatch)
    snapshot(root, {"one.md": "# Warranty\nOriginal fixture terms"})
    build(str(root), str(db), qdrant_url="http://example.invalid")
    snapshot(root, {})
    vectors.fail = failure
    with pytest.raises(RuntimeError, match="injected"):
        build(str(root), str(db), qdrant_url="http://example.invalid")
    assert queue_paths(db) == ["one.md"]
    mail = SQLiteStore(str(db))
    monkeypatch.setattr(server, "mail_store", mail)
    monkeypatch.setattr(server, "mail_vector_store", vectors)
    try:
        # Exercise the actual hybrid ranker and mail lane, not a stubbed search.
        assert server.hybrid_search("warranty", 3, mail, vectors) == []
        assert server.unified_search_with_coverage("warranty", 3, "mail")[0] == []
    finally:
        mail.db.close()
    if readd:
        snapshot(root, {"one.md": "# Warranty\nNew permitted fixture terms"})
    stats = build(str(root), str(db), qdrant_url="http://example.invalid")
    assert stats["added"] == int(readd)
    assert queue_paths(db) == []
    assert_vector_parity(db, vectors)
    assert bool(vectors.points) == readd


def test_changed_mail_rejects_old_and_tampered_vector_payloads(tmp_path, monkeypatch):
    from memory_store import server

    root, db = tmp_path / "export", tmp_path / "mail.sqlite"
    vectors = install_vectors(monkeypatch)
    snapshot(root, {"one.md": "# Warranty\nOld terms"})
    build(str(root), str(db), qdrant_url="http://example.invalid")
    snapshot(root, {"one.md": "# Warranty\nCurrent terms"})
    vectors.fail = "before_delete"
    with pytest.raises(RuntimeError):
        build(str(root), str(db), qdrant_url="http://example.invalid")
    mail = SQLiteStore(str(db))
    monkeypatch.setattr(server, "mail_store", mail)
    monkeypatch.setattr(server, "mail_vector_store", vectors)
    try:
        assert server.hybrid_search("unrelatedzzzz", 3, mail, vectors) == []
        build(str(root), str(db), qdrant_url="http://example.invalid")
        result = server.hybrid_search("unrelatedzzzz", 3, mail, vectors)
        assert len(result) == 1 and result[0]["text"] == "Current terms"
        next(iter(vectors.points.values()))["text"] = "Untrusted old content"
        assert server.hybrid_search("unrelatedzzzz", 3, mail, vectors) == []
    finally:
        mail.db.close()


def test_remote_success_before_local_acknowledgement_retries(tmp_path, monkeypatch):
    root, db = tmp_path / "export", tmp_path / "mail.sqlite"
    vectors = install_vectors(monkeypatch)
    snapshot(root, {"one.md": "# Warranty\nTerms"})
    build(str(root), str(db))  # lexical-only still journals pending vectors
    store = SQLiteStore(str(db))
    store.db.execute(
        "CREATE TRIGGER fail_ack BEFORE DELETE ON mail_vector_pending "
        "BEGIN SELECT RAISE(FAIL, 'injected local acknowledgement failure'); END"
    )
    store.db.commit()
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError, match="injected local"):
        build(str(root), str(db), qdrant_url="http://example.invalid")
    assert vectors.points and queue_paths(db) == ["one.md"]
    store.db.execute("DROP TRIGGER fail_ack")
    store.db.commit()
    store.db.close()
    build(str(root), str(db), qdrant_url="http://example.invalid")
    assert queue_paths(db) == []
    assert_vector_parity(db, vectors)


def test_queue_backfills_preexisting_files_once_and_is_atomic(tmp_path, monkeypatch):
    from memory_store import mail_index

    root, db = tmp_path / "export", tmp_path / "mail.sqlite"
    vectors = install_vectors(monkeypatch)
    snapshot(root, {"one.md": "# Warranty\nTerms"})
    store = SQLiteStore(str(db))
    # Simulate the previous mail builder, with no durable queue yet.
    from memory_store.indexer import Indexer

    Indexer(root / "current/mail", store).scan()
    store.db.close()
    assert build(str(root), str(db), qdrant_url="http://example.invalid")["unchanged"] == 1
    assert_vector_parity(db, vectors)
    assert queue_paths(db) == []
    # No full vector rewrite on an unchanged build (or it would raise).
    vectors.fail = "before_delete"
    build(str(root), str(db), qdrant_url="http://example.invalid")
    assert vectors.fail == "before_delete"
    store = SQLiteStore(str(db))
    mail_index._prepare_vector_queue(store)
    with pytest.raises(RuntimeError):
        with store.db:
            store.db.execute("DELETE FROM files WHERE path='one.md'")
            raise RuntimeError("injected SQLite rollback")
    assert store.file_digest("one.md")
    assert queue_paths(db) == []
    store.db.close()


@pytest.mark.parametrize("collection", ["mail-shared", "mail-privateish", "documents", "", "mail-"])
def test_builder_refuses_non_private_namespace_before_initialization(
    tmp_path, monkeypatch, collection
):
    from memory_store import mail_index

    root, db = tmp_path / "export", tmp_path / "mail.sqlite"
    snapshot(root, {})
    monkeypatch.setattr(mail_index, "SQLiteStore", lambda *a: pytest.fail("SQLite initialized"))
    monkeypatch.setattr(
        mail_index, "MailQdrantStore", lambda *a: pytest.fail("vectors initialized")
    )
    with pytest.raises(ValueError, match="namespace"):
        build(str(root), str(db), qdrant_url="http://example.invalid", collection=collection)
    assert not db.exists()
    assert not (root / ".sync.lock").exists()


@pytest.mark.parametrize(
    "scope", ["QDRANT_COLLECTION", "ARCHIVE_QDRANT_COLLECTION", "DOCUMENTS_QDRANT_COLLECTION"]
)
@pytest.mark.parametrize("collection", ["mail-private", "mail-private-fixture"])
def test_builder_refuses_each_configured_collection_collision(
    tmp_path, monkeypatch, scope, collection
):
    from memory_store import mail_index

    root, db = tmp_path / "export", tmp_path / "mail.sqlite"
    snapshot(root, {})
    monkeypatch.setenv(scope, collection)
    monkeypatch.setattr(
        mail_index, "EmbeddingClient", lambda *a: pytest.fail("embedder initialized")
    )
    monkeypatch.setattr(mail_index, "SQLiteStore", lambda *a: pytest.fail("SQLite initialized"))
    monkeypatch.setattr(
        mail_index, "MailQdrantStore", lambda *a: pytest.fail("vectors initialized")
    )
    with pytest.raises(ValueError, match="collides"):
        build(str(root), str(db), qdrant_url="http://example.invalid", collection=collection)
    assert not db.exists()


def test_mail_qdrant_waits_for_applied_mutations(monkeypatch):
    from memory_store.mail_index import MailQdrantStore
    from memory_store.qdrant import QdrantStore

    calls = []
    monkeypatch.setattr(
        QdrantStore,
        "_request",
        lambda self, method, path, payload=None: calls.append((method, path)) or {},
    )
    vectors = MailQdrantStore("http://example.invalid", "mail-private-fixture")
    vectors.upsert([("chunk", "one.md", "Heading", "Terms", 1)], lambda _: [0.1] * 128)
    vectors.delete_file("one.md")
    assert calls == [
        ("GET", "/collections/mail-private-fixture"),
        ("PUT", "/collections/mail-private-fixture/points?wait=true"),
        ("POST", "/collections/mail-private-fixture/points/delete?wait=true"),
    ]


def test_mail_startup_validation_and_vector_configuration(tmp_path):
    import importlib
    import os
    from unittest.mock import patch

    from memory_store import server

    base = {
        "SQLITE_PATH": str(tmp_path / "index.db"),
        "MAIL_SQLITE_PATH": str(tmp_path / "mail.db"),
        "MAIL_ACCESS_TOKEN": "synthetic-token",
        "QDRANT_URL": "http://example.invalid",
    }
    try:
        for overrides, message in [
            ({"MAIL_SQLITE_PATH": "relative.db"}, "must be absolute"),
            ({"MAIL_SQLITE_PATH": base["SQLITE_PATH"]}, "must be separate"),
            ({"MAIL_QDRANT_COLLECTION": "documents"}, "mail-private namespace"),
            (
                {"MAIL_QDRANT_COLLECTION": "mail-private", "QDRANT_COLLECTION": "mail-private"},
                "mail-private namespace",
            ),
        ]:
            with patch.dict(os.environ, {**base, **overrides}):
                with pytest.raises(ValueError, match=message):
                    importlib.reload(server)
        with patch.dict(os.environ, {**base, "MAIL_QDRANT_COLLECTION": "mail-private-fixture"}):
            module = importlib.reload(server)
            assert module.mail_store is not None
            assert module.mail_vector_store.collection == "mail-private-fixture"
            module.mail_store.close()
    finally:
        importlib.reload(server)
