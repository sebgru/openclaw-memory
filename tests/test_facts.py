"""Synthetic private-fact regressions; no owner data or private paths."""

import copy
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from memory_store import server
from memory_store.facts import (
    REGISTRY,
    FactError,
    FactStore,
    authorized,
    resolve,
    token_from_env,
    validate,
)

AT = datetime(2026, 11, 22, 12, tzinfo=UTC)
STAMP = "2026-11-22T12:00:00Z"
CONFIRM = {"by": "owner", "at": STAMP, "session_ref": "session:synthetic#yes1"}


def claim(n=1, source="document", value="Example Org", date="2026-11-01", **kw):
    fact = dict(
        fact_id=f"f_{n:08d}",
        subject="person:owner",
        predicate="employment.employer",
        item_key=None,
        value=value,
        value_type="string",
        unit=None,
        valid_from=None,
        valid_to=None,
        last_verified=None,
        recorded_at=STAMP,
        privacy_class="employment",
        confidence=0.6,
        status="candidate",
        source_class=source,
        provenance={
            "kind": {
                "owner_stated": "session",
                "conversation": "session",
                "document": "document",
                "mail": "mail",
            }[source],
            "ref": "mh_" + "a" * 64 if source == "mail" else "synthetic/source",
            "evidence_date": date,
        },
    )
    if source == "owner_stated":
        fact.update(
            status="current",
            confidence=0.95,
            confirmation=dict(CONFIRM, at="2026-09-01T12:00:00Z"),
            last_verified="2026-09-01",
        )
    fact.update(kw)
    validate("Fact", fact)
    return fact


def draft(**kw):
    result = dict(
        action="add",
        subject="person:owner",
        predicate="employment.employer",
        value="Example Org",
        value_type="string",
        unit=None,
        valid_from=None,
        privacy_class="employment",
        origin={"session_ref": "session:synthetic#proposal", "audience": "owner_private"},
    )
    result.update(kw)
    return result


@pytest.fixture
def store(tmp_path):
    return FactStore(tmp_path / "private.sqlite", enabled=True, clock=lambda: AT)


def confirmed(store, **kw):
    d = store.draft(draft(**kw))
    return store.confirm(d["draft_id"], CONFIRM)["facts"][0]


def lookup(rows, **kw):
    return resolve(rows, "person:owner", "employment.employer", as_of=STAMP, **kw)


def test_flag_off_and_unused_lane_create_nothing(tmp_path):
    path = tmp_path / "facts.sqlite"
    s = FactStore(path)
    for call in (
        lambda: s.draft(draft()),
        lambda: s.read("person:owner", "employment.employer"),
        lambda: s.confirm("d_12345678", CONFIRM),
        lambda: s.ingest_candidate(claim()),
    ):
        with pytest.raises(FactError):
            call()
    assert not path.exists()
    s.enabled = True
    assert s.read("person:owner", "employment.employer")["state"] == "not_found"
    s.draft(draft())
    assert not path.exists()


def test_capture_confirm_is_one_time_and_owner_stated(store):
    d = store.draft(draft())
    validate("CaptureDraft", d)
    assert d["expires_at"] == "2026-11-23T12:00:00Z"
    assert not Path(store.path).exists()
    result = store.confirm(d["draft_id"], CONFIRM)
    fact = result["facts"][0]
    assert fact["source_class"] == "owner_stated" and fact["status"] == "current"
    assert fact["confirmation"] == CONFIRM
    assert store.read("person:owner", "employment.employer")["current"] == fact
    with pytest.raises(FactError):
        store.confirm(d["draft_id"], CONFIRM)


@pytest.mark.parametrize(
    "confirmation",
    [
        {},
        dict(CONFIRM, by="extractor"),
        dict(CONFIRM, session_ref=""),
        dict(CONFIRM, session_ref="other-session#yes"),
        dict(CONFIRM, session_ref="session:synthetic#proposal"),
        dict(CONFIRM, session_ref="session:synthetic#"),
        dict(CONFIRM, at="2026-11-23T12:00:00Z"),
    ],
)
def test_ambiguous_confirmation_rejected(store, confirmation):
    d = store.draft(draft())
    with pytest.raises(FactError):
        store.confirm(d["draft_id"], confirmation)
    assert not Path(store.path).exists()


def test_expiry_exactly_24_hours(store):
    d = store.draft(draft())
    store.clock = lambda: AT + timedelta(hours=24)
    with pytest.raises(FactError, match="expired"):
        store.confirm(d["draft_id"], CONFIRM)
    assert not Path(store.path).exists()


@pytest.mark.parametrize(
    "value",
    [
        "password: synthetic",
        "api_key=synthetic",
        "sk-syntheticcredential123456789",
        {"note": "Bearer synthetic"},
        {"nested": {"excerpt": "forbidden"}},
        "https://user:synthetic@example.invalid",
    ],
)
def test_secret_screening_before_draft_storage(store, value):
    with pytest.raises(FactError):
        store.draft(draft(value=value))
    assert not store._drafts and not Path(store.path).exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"predicate": "credential.password"},
        {"predicate": "custom.note"},
        {"predicate": "person.age"},
        {"predicate": "health.glucose_now"},
        {"privacy_class": "shareable"},
        {"value_type": "quantity", "unit": "kg"},
        {"origin": {"session_ref": "s#message", "audience": "group"}},
        {"value": {"x": "string expected"}},
        {"will_supersede": []},
    ],
)
def test_schema_and_registry_fail_closed(store, changes):
    with pytest.raises(FactError):
        store.draft(draft(**changes))


def test_replace_symmetric_links_exclusive_end_and_read_only_history(store):
    old = confirmed(store, valid_from="2025-01-01")
    replacement = store.draft(
        draft(action="replace", value="Next Example Org", valid_from="2026-11-01")
    )
    assert replacement["will_supersede"] == [old["fact_id"]]
    new = store.confirm(replacement["draft_id"], CONFIRM)["facts"][0]
    assert new["supersedes"] == [old["fact_id"]]
    current = store.read("person:owner", "employment.employer", include_history=True)
    assert current["current"]["fact_id"] == new["fact_id"]
    historical = current["historical"][0]
    assert historical["status"] == "superseded" and historical["superseded_by"] == new["fact_id"]
    assert historical["valid_to"] == "2026-11-01"
    before = Path(store.path).read_bytes()
    assert (
        store.read("person:owner", "employment.employer", as_of="2026-10-31")["current"]["fact_id"]
        == old["fact_id"]
    )
    assert Path(store.path).read_bytes() == before
    with sqlite3.connect(store.path) as db:
        events = db.execute("SELECT * FROM lifecycle").fetchall()
        assert len(events) == 3
        assert "Example Org" not in str(events)
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM lifecycle")


def test_replace_stale_draft_rolls_back(store):
    confirmed(store, valid_from="2025-01-01")
    a = store.draft(draft(action="replace", value="A", valid_from="2026-01-01"))
    b = store.draft(draft(action="replace", value="B", valid_from="2026-02-01"))
    store.confirm(a["draft_id"], CONFIRM)
    with pytest.raises(FactError, match="changed"):
        store.confirm(b["draft_id"], CONFIRM)
    assert store.read("person:owner", "employment.employer")["current"]["value"] == "A"


def test_unknown_replacement_start_rejected(store):
    confirmed(store)
    with pytest.raises(FactError):
        store.draft(draft(action="replace"))


def test_set_end_leaves_other_member_and_historical_intervals(store):
    common = dict(
        predicate="family.child",
        value_type="entity_ref",
        privacy_class="family",
        valid_from="2020-01-01",
    )
    a = confirmed(store, **common, value="person:child_a", item_key="person:child_a")
    confirmed(store, **common, value="person:child_b", item_key="person:child_b")
    d = store.draft(
        draft(
            **common,
            value="person:child_a",
            item_key="person:child_a",
            action="end",
            target_fact_ids=[a["fact_id"]],
            valid_to="2026-01-01",
        )
    )
    store.confirm(d["draft_id"], CONFIRM)
    assert store.read("person:owner", "family.child", "person:child_a")["state"] == "not_found"
    assert store.read("person:owner", "family.child", "person:child_b")["state"] == "resolved"
    assert (
        store.read("person:owner", "family.child", "person:child_a", as_of="2025-01-01")["state"]
        == "resolved"
    )


def test_retract_purges_payload_and_retains_value_free_tombstone(store):
    fact = confirmed(store, value="SYNTHETIC_FORGET_MARKER")
    pending = store.draft(draft(value="SYNTHETIC_FORGET_MARKER"))
    d = store.draft(draft(action="retract", value=fact["value"], target_fact_ids=[fact["fact_id"]]))
    store.confirm(d["draft_id"], CONFIRM)
    assert store.read("person:owner", "employment.employer")["state"] == "not_found"
    assert b"SYNTHETIC_FORGET_MARKER" not in Path(store.path).read_bytes()
    assert pending["draft_id"] not in store._drafts
    with sqlite3.connect(store.path) as db:
        tombstone = json.loads(db.execute("SELECT payload FROM tombstones").fetchone()[0])
        validate("RetractionTombstone", tombstone)
        assert "value" not in tombstone


def test_ingest_cannot_promote_or_overwrite(store):
    f = claim()
    store.ingest_candidate(f)
    with pytest.raises(FactError):
        store.ingest_candidate(dict(f, value="changed"))
    with pytest.raises(FactError):
        store.ingest_candidate(claim(2, source="owner_stated"))
    assert store.read("person:owner", "employment.employer")["current"]["status"] == "candidate"


def test_owner_over_old_document_new_mail_and_new_credible_document():
    owner = claim(1, "owner_stated", "Org A")
    old = claim(2, value="Org B", date="2023-01-01")
    mail = claim(3, "mail", "Org B", date="2026-11-01")
    result = lookup([owner, old, mail])
    assert result["current"]["value"] == "Org A" and result["source_rank"] == 0
    assert result["state"] == "resolved" and "noncurrent_claim" in result["disclosures"]
    newer = claim(4, value="Org C", date="2026-11-15")
    result = lookup([owner, old, newer])
    assert result["current"]["value"] == "Org A" and result["state"] == "conflict"


def test_document_precedence_and_unknown_and_equal_date_conflicts():
    a = claim(1, value="Org A", date="2024-01-01")
    b = claim(2, value="Org B", date="2026-11-01")
    result = lookup([a, b])
    assert result["current"]["value"] == "Org B"
    assert "unreviewed" in result["disclosures"]
    for c in [
        claim(3, value="Org C", date="2026-11-01", confidence=0.4),
        claim(4, value="Org D", date=None),
    ]:
        result = lookup([a, b, c])
        assert result["state"] == "conflict" and "current" not in result


def test_recent_conversation_and_expiry_do_not_mutate():
    a = claim(1, value="Org A", date="2023-01-01")
    b = claim(2, "conversation", value="Org B", date="2026-10-05")
    before = copy.deepcopy([a, b])
    result = lookup([a, b])
    assert result["source_rank"] == 1 and result["current"]["value"] == "Org B"
    result = resolve([a, b], "person:owner", "employment.employer", as_of="2026-12-10")
    assert result["source_rank"] == 2 and result["current"]["value"] == "Org B"
    assert [a, b] == before


def test_seeded_owner_conflict_not_resolved_by_confidence():
    result = lookup(
        [claim(1, "owner_stated", "Org A"), claim(2, "owner_stated", "Org B", confidence=1)]
    )
    assert result["state"] == "conflict" and "current" not in result


def weight(n, date, value, source="conversation"):
    return claim(
        n,
        source,
        value,
        date,
        predicate="body.weight",
        value_type="quantity",
        unit="kg",
        privacy_class="health",
        valid_from=date,
    )


def test_series_latest_beats_old_owner_and_same_day_conflicts():
    rows = [weight(1, "2021-01-01", 80, "owner_stated"), weight(2, "2026-11-18", 90)]
    r = resolve(rows, "person:owner", "body.weight", as_of=STAMP, include_history=True)
    assert r["current"]["value"] == 90 and len(r["historical"]) == 1
    assert "supersedes" not in rows[1]
    rows.append(weight(3, "2026-11-18", 91))
    r = resolve(rows, "person:owner", "body.weight", as_of=STAMP)
    assert r["state"] == "conflict" and "current" not in r
    r = resolve(rows[:2], "person:owner", "body.weight", as_of="2026-12-20")
    assert r["state"] == "resolved_stale"


def test_static_dob_recency_does_not_choose_truth():
    rows = [
        claim(
            i,
            value=v,
            date=d,
            predicate="identity.date_of_birth",
            value_type="date",
            privacy_class="identity",
        )
        for i, v, d in [(1, "1980-01-01", "2020-01-01"), (2, "1981-01-01", "2026-01-01")]
    ]
    assert (
        resolve(rows, "person:owner", "identity.date_of_birth", as_of=STAMP)["state"] == "conflict"
    )


def test_low_confidence_mail_only_missing_and_unknown_disclosures():
    assert lookup([])["state"] == "not_found"
    assert lookup([claim(confidence=0.3)])["state"] == "unverified_only"
    r = lookup([claim(source="mail")])
    assert r["state"] == "unverified_only" and "mail_only" in r["disclosures"]
    r = lookup([claim(date=None)])
    assert {"unknown_source_date", "uncertain_validity", "unreviewed"} <= set(r["disclosures"])


def test_token_env_file_and_constant_time_comparison(tmp_path):
    path = tmp_path / "token"
    path.write_text("synthetic-only\n")
    assert token_from_env({"MEMORY_FACTS_WRITE_TOKEN_FILE": str(path)}) == "synthetic-only"
    assert token_from_env({"MEMORY_FACTS_WRITE_TOKEN_FILE": str(path / "missing")}) == ""
    assert authorized("Bearer synthetic-only", "synthetic-only")
    assert not authorized("Bearer wrong", "synthetic-only")
    assert not authorized(None, "")


@pytest.fixture
def api(store):
    with (
        patch.object(server, "FACTS_ENABLED", True),
        patch.object(server, "FACTS_TOKEN", "synthetic-only"),
        patch.object(server, "fact_store", store),
    ):
        http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        thread = Thread(target=http.serve_forever, daemon=True)
        thread.start()

        def request(method, path, payload=None, token="synthetic-only"):
            c = HTTPConnection(*http.server_address)
            headers = {"Authorization": "Bearer " + token} if token is not None else {}
            c.request(method, path, json.dumps(payload) if payload is not None else None, headers)
            r = c.getresponse()
            result = r.status, json.loads(r.read())
            c.close()
            return result

        yield request
        http.shutdown()
        http.server_close()
        thread.join()


@pytest.mark.parametrize(
    "path,method",
    [
        ("/facts/current?subject=person:owner&predicate=employment.employer", "GET"),
        ("/facts/drafts", "POST"),
        ("/facts/confirm", "POST"),
        ("/facts/end", "POST"),
        ("/facts/retract", "POST"),
        (
            "/unified/search?q=example&fact_subject=person:owner&fact_predicate=employment.employer",
            "GET",
        ),
    ],
)
def test_api_missing_wrong_right_token_and_fail_closed(api, path, method):
    assert api(method, path, draft(), token=None)[0] == 401
    assert api(method, path, draft(), token="wrong")[0] == 401
    assert api(method, path, draft())[0] not in {401, 503}
    with patch.object(server, "FACTS_TOKEN", ""):
        assert api(method, path, draft())[0] == 503


def test_api_capture_and_unified_block_ahead_of_evidence(api):
    status, d = api("POST", "/facts/drafts", draft())
    assert status == 200
    assert (
        api("POST", "/facts/confirm", {"draft_id": d["draft_id"], "confirmation": CONFIRM})[0]
        == 200
    )
    with patch.object(server, "unified_search_with_coverage", return_value=([], [], {}, {})):
        status, payload = api(
            "GET",
            "/unified/search?q=example&fact_subject=person:owner&fact_predicate=employment.employer",
        )
    assert status == 200 and list(payload)[0] == "current_state"
    assert payload["current_state"][0]["current"]["value"] == "Example Org"


def test_flag_off_api_surface_unchanged(api, store):
    with patch.object(server, "FACTS_ENABLED", False):
        for path in ["/facts/drafts", "/facts/confirm", "/facts/end", "/facts/retract"]:
            assert api("POST", path, draft())[0] == 404
        assert api("GET", "/facts/current")[0] == 404
        with patch.object(server, "unified_search_with_coverage", return_value=([], [], {}, {})):
            status, data = api(
                "GET",
                "/unified/search?q=example&fact_subject=person:owner&fact_predicate=employment.employer",
                token=None,
            )
            assert status == 200 and data == {"results": [], "coverage": {}}
    assert not Path(store.path).exists()


def test_private_fact_access_logs_are_redacted(api, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="memory_store"):
        api(
            "GET",
            "/facts/current?subject=person:owner&predicate=employment.employer&as_of=PRIVATE_MARKER",
        )
    assert "PRIVATE_MARKER" not in caplog.text


def test_schema_scrubbed_generic_only():
    from memory_store.facts import _SCHEMA

    text = json.dumps(_SCHEMA)
    assert "excerpt" not in text
    for private in ("/private/workspace", "PRIVATE_OWNER_SENTINEL", "migration"):
        assert private not in text
    assert "pref.timezone" in REGISTRY


def test_derived_age_calendar_boundaries_leap_day_and_privacy():
    dob = claim(
        1,
        "owner_stated",
        "2020-02-29",
        predicate="identity.date_of_birth",
        value_type="date",
        subject="person:child_a",
        privacy_class="family",
    )
    for date, expected in [("2026-02-28", 5), ("2026-03-01", 6)]:
        r = resolve([dob], "person:child_a", "person.age", as_of=date)
        assert r["derived_value"] == expected and r["derived_privacy_class"] == "family"
        assert r["derived_from"] == [dob["fact_id"]] and "derived" in r["disclosures"]
    other = dict(dob, fact_id="f_00000002", value="2021-02-28")
    r = resolve([dob, other], "person:child_a", "person.age", as_of=STAMP)
    assert r["state"] == "conflict" and "derived_value" not in r
    assert resolve([], "person:child_a", "person.age", as_of=STAMP)["state"] == "not_found"


def test_derived_age_owner_timezone():
    dob = claim(
        1,
        "owner_stated",
        "2000-11-23",
        predicate="identity.date_of_birth",
        value_type="date",
        privacy_class="identity",
    )
    r = resolve(
        [dob], "person:owner", "person.age", as_of="2026-11-22T23:30:00Z", timezone="Europe/Berlin"
    )
    assert r["derived_value"] == 26


def test_capture_conflicting_owner_claims_and_retraction_reconcile(store):
    first = confirmed(store)
    second = confirmed(store, value="Another Synthetic Org")
    assert second["status"] == "conflict" and first["fact_id"] in second["conflicts_with"]
    r = store.read("person:owner", "employment.employer")
    assert r["state"] == "conflict" and "current" not in r
    d = store.draft(
        draft(action="retract", value=second["value"], target_fact_ids=[second["fact_id"]])
    )
    store.confirm(d["draft_id"], CONFIRM)
    assert store.read("person:owner", "employment.employer")["current"]["status"] == "current"


def test_forget_historical_fact_and_suppress_reextraction(store):
    candidate = claim()
    store.ingest_candidate(candidate)
    d = store.draft(draft(action="retract", target_fact_ids=[candidate["fact_id"]]))
    store.confirm(d["draft_id"], CONFIRM)
    with pytest.raises(FactError, match="suppressed"):
        store.ingest_candidate(dict(candidate, fact_id="f_00000002"))
    old = confirmed(store, valid_from="2025-01-01")
    d = store.draft(draft(action="replace", value="New Synthetic Org", valid_from="2026-01-01"))
    store.confirm(d["draft_id"], CONFIRM)
    d = store.draft(draft(action="retract", value=old["value"], target_fact_ids=[old["fact_id"]]))
    store.confirm(d["draft_id"], CONFIRM)
    r = store.read("person:owner", "employment.employer", include_history=True)
    assert not r.get("historical")


def test_retract_readback_cannot_target_other_value(store):
    old = confirmed(store)
    with pytest.raises(FactError, match="read-back"):
        store.draft(
            draft(action="retract", value="Not the target value", target_fact_ids=[old["fact_id"]])
        )


def test_two_vehicle_keys_no_sale_inferred_and_staleness():
    rows = [
        claim(
            i,
            value={"vehicle": key},
            date=date,
            predicate="vehicle.owned",
            value_type="structured",
            item_key=key,
            privacy_class="financial",
            last_verified=date,
        )
        for i, key, date in [(1, "car_a", "2019-01-01"), (2, "car_b", "2026-01-01")]
    ]
    a = resolve(rows, "person:owner", "vehicle.owned", "car_a", STAMP)
    b = resolve(rows, "person:owner", "vehicle.owned", "car_b", STAMP)
    assert a["state"] == "resolved_stale" and b["state"] == "resolved"
    assert a["current"]["valid_to"] is None and b["current"]["valid_to"] is None


def test_api_backend_unavailable_is_not_not_found(api, store):
    with patch.object(store, "read", side_effect=sqlite3.OperationalError("synthetic")):
        status, body = api(
            "GET", "/facts/current?subject=person:owner&predicate=employment.employer"
        )
        assert status == 503 and body["state"] == "unavailable"
        validate("ResolvedFact", body)
        with patch.object(server, "unified_search_with_coverage", return_value=([], [], {}, {})):
            status, body = api(
                "GET",
                "/unified/search?q=example&fact_subject=person:owner&fact_predicate=employment.employer",
            )
            assert status == 200 and body["current_state"][0]["state"] == "unavailable"


def test_startup_flag_off_and_missing_token_no_db_creation(tmp_path):
    import os
    import subprocess

    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        "SQLITE_PATH": str(tmp_path / "index.db"),
        "DOCUMENTS_ROOT": "",
        "ARCHIVE_ROOT": "",
        "MEMORY_FACTS_SQLITE_PATH": str(tmp_path / "facts.db"),
        "MEMORY_FACTS_WRITE_TOKEN": "",
        "MEMORY_FACTS_WRITE_TOKEN_FILE": "",
        "QDRANT_URL": "",
    }
    for enabled in ("false", "true"):
        result = subprocess.run(
            [sys.executable, "-c", "from memory_store import server; print(server.FACTS_ENABLED)"],
            env={**env, "MEMORY_FACTS_ENABLED": enabled},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert not (tmp_path / "facts.db").exists()
        assert ("fail closed" in result.stderr) == (enabled == "true")


@pytest.mark.parametrize(
    "changes",
    [
        {
            "predicate": "family.child",
            "value": "person:child_a",
            "value_type": "entity_ref",
            "privacy_class": "family",
        },
        {"item_key": "unexpected"},
        {
            "predicate": "body.weight",
            "value": 80,
            "value_type": "quantity",
            "unit": "kg",
            "privacy_class": "health",
        },
        {"valid_from": "2026-01-02", "valid_to": "2026-01-01"},
        {"origin": {"session_ref": "missing-message", "audience": "owner_private"}},
        {"target_fact_ids": ["f_00000001"]},
    ],
)
def test_capture_semantic_failure_matrix(store, changes):
    with pytest.raises(FactError):
        store.draft(draft(**changes))


def test_invalid_draft_type_and_pending_limit(store):
    with pytest.raises(FactError):
        store.draft([])
    for i in range(100):
        store.draft(draft(value=f"Synthetic {i}"))
    with pytest.raises(FactError, match="limit"):
        store.draft(draft())
    store.clock = lambda: AT + timedelta(days=1)
    assert store.draft(draft())["draft_id"]
    assert len(store._drafts) == 1


def test_unknown_targets_and_no_existing_replace(store):
    with pytest.raises(FactError, match="no target"):
        store.draft(draft(action="replace", valid_from="2026-01-01"))
    old = confirmed(store, valid_from="2020-01-01")
    with pytest.raises(FactError, match="existing"):
        store.draft(
            draft(
                action="replace",
                predicate="pref.language",
                privacy_class="shareable",
                valid_from="2026-01-01",
            )
        )
    with pytest.raises(FactError, match="target"):
        store.draft(draft(action="retract", target_fact_ids=["f_99999999"]))
    with pytest.raises(FactError, match="transition"):
        store.draft(draft(action="replace", valid_from="2019-01-01"))
    with pytest.raises(FactError, match="duplicate"):
        confirmed(store, valid_from="2020-01-01")
    assert store.read("person:owner", "employment.employer")["current"]["fact_id"] == old["fact_id"]


def test_series_capture_preserves_observations_and_refuses_end(store):
    common = dict(predicate="body.weight", value_type="quantity", unit="kg", privacy_class="health")
    a = confirmed(store, **common, value=80, valid_from="2020-01-01")
    confirmed(store, **common, value=90, valid_from="2026-11-20")
    with pytest.raises(FactError, match="state/set"):
        store.draft(
            draft(
                **common,
                value=80,
                valid_from="2020-01-01",
                action="end",
                target_fact_ids=[a["fact_id"]],
                valid_to="2026-01-01",
            )
        )
    result = store.read("person:owner", "body.weight", include_history=True)
    assert result["current"]["value"] == 90 and len(result["historical"]) == 1


def test_candidate_then_owner_add_and_changed_replacement_target_set(store):
    store.ingest_candidate(claim())
    first = confirmed(store)
    replacement = store.draft(draft(action="replace", value="Replacement", valid_from="2026-01-01"))
    confirmed(store, value="Additional conflict")
    with pytest.raises(FactError, match="replacement changed"):
        store.confirm(replacement["draft_id"], CONFIRM)
    assert store.read("person:owner", "employment.employer")["state"] == "conflict"
    assert first["status"] == "current"


def test_bad_query_type_and_unknown_predicate():
    from memory_store.facts import instant

    with pytest.raises(FactError):
        instant(1)
    with pytest.raises(FactError):
        resolve([], "person:owner", "custom.unknown")
    r = resolve(
        [claim(), claim(2, "mail", "Older org", date="2020-01-01")],
        "person:owner",
        "employment.employer",
    )
    assert r["current"]["value"] == "Example Org"


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/facts/drafts", []),
        ("/facts/drafts", "x" * 17000),
        ("/facts/confirm", {"draft_id": [], "confirmation": CONFIRM}),
        ("/facts/confirm", {}),
        ("/facts/end", draft(action="add")),
        ("/facts/retract", draft(action="add")),
    ],
)
def test_api_malformed_requests_are_generic(api, path, payload):
    status, result = api("POST", path, payload)
    assert status == 400 and "error" in result


def test_api_empty_missing_params_and_write_unavailable(api, store):
    assert api("POST", "/facts/drafts")[0] == 400
    assert api("GET", "/facts/current")[0] == 400
    assert api("GET", "/unified/search?q=x&fact_subject=person:owner")[0] == 400
    with patch.object(store, "draft", side_effect=sqlite3.OperationalError("PRIVATE_MARKER")):
        status, result = api("POST", "/facts/drafts", draft())
        assert status == 503 and "PRIVATE_MARKER" not in str(result)


def test_token_inline_and_wrong_scheme():
    assert token_from_env({"MEMORY_FACTS_WRITE_TOKEN": " synthetic-only "}) == "synthetic-only"
    assert not authorized("Basic synthetic-only", "synthetic-only")


def test_same_index_and_fact_path_refused_at_startup(tmp_path):
    import os
    import subprocess

    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        "SQLITE_PATH": str(tmp_path / "same.db"),
        "DOCUMENTS_ROOT": "",
        "ARCHIVE_ROOT": "",
        "MEMORY_FACTS_SQLITE_PATH": str(tmp_path / "same.db"),
        "MEMORY_FACTS_ENABLED": "true",
        "QDRANT_URL": "",
    }
    result = subprocess.run(
        [sys.executable, "-c", "from memory_store import server"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "must be separate" in result.stderr


def test_replacement_never_closes_weaker_claim_intervals(store):
    candidate = claim(1, value="Document-only older org")
    store.ingest_candidate(candidate)
    owner = confirmed(store, valid_from="2025-01-01")
    d = store.draft(draft(action="replace", value="Next Org", valid_from="2026-01-01"))
    assert d["will_supersede"] == [owner["fact_id"]]
    store.confirm(d["draft_id"], CONFIRM)
    with sqlite3.connect(store.path) as db:
        weaker = json.loads(
            db.execute(
                "SELECT payload FROM facts WHERE fact_id=?", (candidate["fact_id"],)
            ).fetchone()[0]
        )
    assert weaker["status"] == "candidate" and weaker["valid_to"] is None
    assert "superseded_by" not in weaker


def test_internal_lifecycle_cannot_mutate_claim_value(store):
    fact = confirmed(store)
    with pytest.raises(FactError, match="immutable"), store._db() as db:
        store._save(db, dict(fact, value="Illegitimate mutation"), "end", "current")
    assert store.read("person:owner", "employment.employer")["current"]["value"] == fact["value"]
