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
    root = tmp_path / "export"
    with pytest.raises(ValueError, match="completed"):
        build(str(root), str(tmp_path / "mail.db"))
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

        def upsert(self, records, embed):
            assert records
            calls.append("upsert")

    monkeypatch.setattr(mail_index, "QdrantStore", Vectors)
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
            "mail-fixture",
        ],
    )
    mail_index.main()
    assert json.loads(capsys.readouterr().out)["added"] == 1
    assert calls == [("http://example.invalid", "mail-fixture", 128), "upsert"]
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
            ],
        )
        result, warnings, coverage, degraded = server.unified_search_with_coverage(
            "warranty", 3, "mail"
        )
        assert len(result) == 1
        assert result[0]["mail"]["date"] == "2026-01-01"
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
