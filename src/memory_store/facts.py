"""Private, opt-in fact capture. No database or draft persistence at import time."""

import copy
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import unicodedata
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from pathlib import Path
from zoneinfo import ZoneInfo

_SCHEMA = json.loads(files("memory_store").joinpath("fact_schema.json").read_text())


class FactError(ValueError):
    """Invalid capture, lifecycle operation or lookup (messages never echo values)."""


def validate(kind, payload):
    # Keep optional feature dependencies off the disabled server import path.
    from jsonschema import Draft202012Validator, FormatChecker

    validator = Draft202012Validator(
        {"$ref": f"#/$defs/{kind}", "$defs": _SCHEMA["$defs"]},
        format_checker=FormatChecker(),
    )
    if not validator.is_valid(payload):
        raise FactError("invalid " + kind)


def instant(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2}))?",
        value,
    ):
        raise FactError("invalid date")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
        return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)
    except ValueError:
        raise FactError("invalid date") from None


def now():
    return datetime.now(UTC)


def stamp(value):
    return value.isoformat().replace("+00:00", "Z")


# temporal class, type, unit, minimum privacy, freshness in days
REGISTRY = {
    "identity.full_name": ("state", "string", None, "personal", None),
    "identity.date_of_birth": ("static", "date", None, "identity", None),
    "identity.sex": ("state", "string", None, "health", None),
    "person.age_stated": ("series", "quantity", "a", "family", None),
    "body.height": ("state", "quantity", "cm", "health", 3650),
    "body.weight": ("series", "quantity", "kg", "health", 14),
    "housing.address": ("state", "address", None, "personal", 1095),
    "family.spouse": ("set", "entity_ref", None, "family", None),
    "family.child": ("set", "entity_ref", None, "family", None),
    "education.school": ("state", "entity_ref", None, "family", 365),
    "employment.employer": ("state", "string", None, "employment", 730),
    "vehicle.owned": ("set", "structured", None, "financial", 1095),
    "insurance.policy": ("set", "structured", None, "financial", 730),
    "health.condition": ("set", "string", None, "health", None),
    "health.therapy": ("set", "string", None, "health", 730),
    "health.device": ("set", "string", None, "health", 730),
    "contact.mobile_number": ("state", "string", None, "personal", 1095),
    "pref.language": ("state", "string", None, "shareable", None),
    "pref.timezone": ("state", "string", None, "shareable", None),
}


def screen(payload):
    """Conservative defense in depth, not a universal semantic secret detector."""
    text = "".join(
        char
        for char in unicodedata.normalize("NFKC", json.dumps(payload, ensure_ascii=False))
        if unicodedata.category(char) != "Cf"
    )
    patterns = (
        r"(?i)pass(?:word|phrase)|passwd|credential|api[_ -]?key|\bsecret\b|\btoken\b|\bpin\b|\botp\b|\bcvv\b|kennwort|zugangsdaten|access[_ -]?token|bearer\s|authorization\s*:\s*basic|private[_ -]?key|client[_ -]?secret|recovery[_ -]?(?:code|phrase)|seed[_ -]?phrase|пароль|passwort",
        r"\b[a-fA-F0-9]{32,}\b",
        r"\b[A-Z2-7]{16,}\b",
        r"\b(?:sk-|ghp_|github_pat_|xox[baprs]-|AKIA)[A-Za-z0-9_-]+",
        r"-----BEGIN .*PRIVATE KEY-----",
        r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        r"://[^\s/:]+:[^\s/@]+@",
        # Opaque long credential-like strings: prefer rejection over silent retention.
        r"\b(?=[A-Za-z0-9_+/=-]{32,}\b)(?=[A-Za-z0-9_+/=-]*[A-Z])(?=[A-Za-z0-9_+/=-]*[a-z])(?=[A-Za-z0-9_+/=-]*\d)[A-Za-z0-9_+/=-]{32,}",
    )
    if any(re.search(pattern, text) for pattern in patterns):
        raise FactError("credential-like capture rejected")

    def walk(value):
        if isinstance(value, dict):
            if "excerpt" in value:
                raise FactError("excerpts are forbidden")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)


def screen_retained_fact(fact):
    """Screen every stored field, exempting only schema-validated digest locations."""
    payload = copy.deepcopy(fact)
    for provenance in [payload["provenance"], *payload.get("evidence", [])]:
        provenance.pop("content_sha256", None)
    screen(payload)


def semantics(fact):
    predicate = fact["predicate"]
    if predicate not in REGISTRY:
        raise FactError(
            "predicate requires registry review; derived/volatile facts cannot be stored"
        )
    temporal, value_type, unit, privacy, _ = REGISTRY[predicate]
    if fact["value_type"] != value_type or fact["unit"] != unit:
        raise FactError("registry type or unit mismatch")
    required_privacy = (
        "family" if fact["subject"].startswith(("person:child", "person:spouse")) else privacy
    )
    if fact["privacy_class"] not in {required_privacy, "restricted"}:
        raise FactError("privacy class below registry/subject policy")
    if temporal == "set" and not fact.get("item_key"):
        raise FactError("set membership requires item_key")
    if temporal != "set" and fact.get("item_key") is not None:
        raise FactError("item_key only applies to sets")
    if temporal == "series" and not fact["valid_from"]:
        raise FactError("series requires observation date")
    for bound in ("valid_from", "valid_to"):
        if fact.get(bound) is not None:
            instant(fact[bound])
    if fact.get("valid_to") and fact["valid_from"]:
        if instant(fact["valid_to"]) <= instant(fact["valid_from"]):
            raise FactError("invalid exclusive interval")


def slot(fact):
    return fact["subject"], fact["predicate"], fact.get("item_key")


def resolve(
    facts,
    subject,
    predicate,
    item_key=None,
    as_of=None,
    include_history=False,
    recency_days=60,
    timezone="UTC",
):
    """Read-only deterministic precedence. No rank or lifecycle writes."""
    if predicate == "person.age":
        resolved = resolve(
            facts,
            subject,
            "identity.date_of_birth",
            item_key,
            as_of,
            include_history,
            recency_days,
            timezone,
        )
        resolved["predicate"] = predicate
        current = resolved.pop("current", None)
        resolved["disclosures"] = sorted(set(resolved["disclosures"]) | {"derived"})
        if current:
            resolved["derived_from"] = [current["fact_id"]]
            resolved["derived_privacy_class"] = current["privacy_class"]
            if resolved["state"] != "conflict":
                birthday = datetime.fromisoformat(current["value"]).date()
                at_date = instant(as_of or stamp(now())).astimezone(ZoneInfo(timezone)).date()
                # Calendar comparison gives March 1 for leap-day DOB in non-leap years.
                resolved["derived_value"] = (
                    at_date.year
                    - birthday.year
                    - ((at_date.month, at_date.day) < (birthday.month, birthday.day))
                )
        validate("ResolvedFact", resolved)
        return resolved
    if predicate not in REGISTRY:
        raise FactError("unknown predicate")
    as_of = as_of or stamp(now())
    at = instant(as_of)
    temporal, _, _, _, max_age = REGISTRY[predicate]
    result = dict(
        subject=subject,
        predicate=predicate,
        item_key=item_key,
        as_of=as_of,
        state="not_found",
        disclosures=[],
    )
    flags = set()
    rows = [
        copy.deepcopy(f)
        for f in facts
        if slot(f) == (subject, predicate, item_key) and f["status"] != "retracted"
    ]
    eligible, history = [], []
    for fact in rows:
        start, end = fact["valid_from"], fact["valid_to"]
        if (start and instant(start) > at) or (end and instant(end) <= at):
            history.append(fact)
        else:
            eligible.append(fact)
    if temporal == "series" and eligible:
        latest = max(instant(f["valid_from"]) for f in eligible)
        history += [f for f in eligible if instant(f["valid_from"]) < latest]
        eligible = [f for f in eligible if instant(f["valid_from"]) == latest]
    strong = [f for f in eligible if f["confidence"] >= 0.4 and f["source_class"] != "mail"]
    dated_band = [
        instant(f["provenance"]["evidence_date"])
        for f in strong
        if f["source_class"] in {"document", "conversation"} and f["provenance"]["evidence_date"]
    ]
    newest = max(dated_band) if dated_band else None

    def rank(f):
        source, date = f["source_class"], f["provenance"]["evidence_date"]
        if source == "owner_stated":
            return 0
        if (
            source == "conversation"
            and date
            and timedelta(0) <= at - instant(date) <= timedelta(days=recency_days)
        ):
            return 1
        return 2 if date and instant(date) == newest else 3

    if not strong:
        if eligible:
            result["state"] = "unverified_only"
            result["alternatives"] = sorted(eligible, key=lambda f: f["fact_id"])
            flags.add("unreviewed")
            if all(f["source_class"] == "mail" for f in eligible):
                flags.add("mail_only")
                result.update(source_rank=4, source_tier="mail")
            if any(f["confidence"] < 0.4 for f in eligible):
                flags.add("low_confidence")
    else:
        best = min(rank(f) for f in strong)
        leaders = [f for f in strong if rank(f) == best]
        # Unknown-dated documents cannot be silently discarded as old evidence.
        if best in {2, 3}:
            leaders += [
                f
                for f in strong
                if rank(f) != best
                and f["source_class"] in {"document", "conversation"}
                and not f["provenance"]["evidence_date"]
            ]
        if temporal == "static":
            leaders = [
                f
                for f in strong
                if rank(f) == best or f["source_class"] == leaders[0]["source_class"]
            ]
        leaders.sort(key=lambda f: f["fact_id"])
        values = {json.dumps(f["value"], sort_keys=True) for f in leaders}
        result["source_rank"] = best
        result["source_tier"] = (
            "T0_owner_asserted",
            "T1_recent",
            "document_newer",
            "document_older",
            "mail",
        )[best]
        if len(values) > 1:
            result["state"] = "conflict"
            result["alternatives"] = eligible
            flags.add("conflict")
        else:
            chosen = leaders[0]
            result["current"] = chosen
            result["state"] = "resolved"
            alternatives = [f for f in eligible if f["fact_id"] != chosen["fact_id"]]
            result["alternatives"] = alternatives
            verified = (
                chosen["last_verified"] or chosen["valid_from"]
                if temporal == "series"
                else chosen["last_verified"]
            )
            if max_age and verified and (at - instant(verified)).days > max_age:
                result["state"] = "resolved_stale"
                flags.add("stale")
            elif max_age and not verified:
                flags.add("uncertain_validity")
            confirmed = chosen.get("confirmation", {}).get("at")
            for other in alternatives:
                date = other["provenance"]["evidence_date"]
                if other["value"] != chosen["value"]:
                    if other["source_class"] == "mail":
                        flags.add("noncurrent_claim")
                    elif (
                        best == 0
                        and date
                        and confirmed
                        and instant(date) > instant(confirmed)
                        and other["confidence"] >= 0.4
                    ):
                        result["state"] = "conflict"
                        flags.add("conflict")
                    else:
                        flags.add("historical")
        if any(f["status"] == "candidate" for f in leaders):
            flags.add("unreviewed")
    for f in eligible:
        if f["provenance"]["evidence_date"] is None:
            flags.add("unknown_source_date")
        if f["valid_from"] is None:
            flags.add("uncertain_validity")
    if include_history:
        result["historical"] = history
        if history:
            flags.add("historical")
    result["disclosures"] = sorted(flags)
    validate("ResolvedFact", result)
    return result


class FactStore:
    """One private store per instance. Drafts are bounded, volatile session state."""

    def __init__(self, path, enabled=False, clock=now, recency_days=60, timezone="UTC"):
        self.path, self.enabled, self.clock = str(path), enabled, clock
        self.recency_days = recency_days
        self.timezone = timezone
        ZoneInfo(timezone)
        self._lock = threading.RLock()
        self._drafts = {}
        self._schema_identity = None

    def _enabled(self):
        if not self.enabled:
            raise FactError("facts disabled")

    def _prune_drafts(self):
        with self._lock:
            at = self.clock()
            self._drafts = {
                key: value
                for key, value in self._drafts.items()
                if instant(value["expires_at"]) > at
            }

    @contextmanager
    def _db(self):
        self._enabled()
        # No connection/file until an enabled operation needs it.
        with self._lock:
            descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(descriptor)
            db = sqlite3.connect(self.path, timeout=2)
            db.row_factory = sqlite3.Row
            try:
                db.execute("PRAGMA secure_delete=ON")
                stat = os.stat(self.path)
                identity = (stat.st_dev, stat.st_ino)
                initialize = self._schema_identity != identity
                if initialize:
                    db.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                  fact_id TEXT PRIMARY KEY, subject TEXT NOT NULL, predicate TEXT NOT NULL,
                  item_key TEXT, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS fact_slot ON facts(subject,predicate,item_key);
                CREATE TABLE IF NOT EXISTS lifecycle (
                  seq INTEGER PRIMARY KEY, at TEXT NOT NULL, actor TEXT NOT NULL,
                  fact_id TEXT NOT NULL, action TEXT NOT NULL, prior_status TEXT,
                  new_status TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS lifecycle_no_update BEFORE UPDATE ON lifecycle
                  BEGIN SELECT RAISE(ABORT,'append only'); END;
                CREATE TRIGGER IF NOT EXISTS lifecycle_no_delete BEFORE DELETE ON lifecycle
                  BEGIN SELECT RAISE(ABORT,'append only'); END;
                CREATE TABLE IF NOT EXISTS tombstones (fact_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS lifecycle_fact_id ON lifecycle(fact_id);
                """)
                db.execute("BEGIN IMMEDIATE")
                if initialize:
                    db.execute(
                        "INSERT OR IGNORE INTO metadata VALUES('suppression_salt', ?)",
                        (secrets.token_hex(32),),
                    )
                yield db
                db.commit()
                self._schema_identity = identity
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()

    @staticmethod
    def _rows(db, subject, predicate, item_key=None):
        return [
            json.loads(row[0])
            for row in db.execute(
                "SELECT payload FROM facts WHERE subject=? AND predicate=? AND item_key IS ?",
                (subject, predicate, item_key),
            )
        ]

    def read(self, subject, predicate, item_key=None, as_of=None, include_history=False):
        self._enabled()
        self._prune_drafts()
        # Reads of an unused lane do not create an empty database.
        if not Path(self.path).exists():
            rows = []
        else:
            db = sqlite3.connect(
                Path(self.path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2
            )
            try:
                rows = self._rows(
                    db,
                    subject,
                    "identity.date_of_birth" if predicate == "person.age" else predicate,
                    item_key,
                )
            finally:
                db.close()
        return resolve(
            rows,
            subject,
            predicate,
            item_key,
            as_of or stamp(self.clock()),
            include_history,
            self.recency_days,
            self.timezone,
        )

    @staticmethod
    def _suppression_key(db, fact):
        salt = db.execute("SELECT value FROM metadata WHERE name='suppression_salt'").fetchone()[0]
        pointer = json.dumps([*slot(fact), fact["provenance"]["kind"], fact["provenance"]["ref"]])
        return hmac.new(salt.encode(), pointer.encode(), hashlib.sha256).hexdigest()

    def _audit(self, db, fact_id, action, prior, status):
        db.execute(
            "INSERT INTO lifecycle(at,actor,fact_id,action,prior_status,new_status) VALUES(?,?,?,?,?,?)",
            (stamp(self.clock()), "openclaw_instance", fact_id, action, prior, status),
        )

    @staticmethod
    def _id_reserved(db, fact_id):
        return (
            db.execute(
                "SELECT 1 FROM facts WHERE fact_id=? UNION ALL "
                "SELECT 1 FROM tombstones WHERE fact_id=? UNION ALL "
                "SELECT 1 FROM lifecycle WHERE fact_id=? LIMIT 1",
                (fact_id, fact_id, fact_id),
            ).fetchone()
            is not None
        )

    def _save(self, db, fact, action, prior=None):
        validate("Fact", fact)
        screen_retained_fact(fact)
        previous = db.execute(
            "SELECT payload FROM facts WHERE fact_id=?", (fact["fact_id"],)
        ).fetchone()
        if previous:
            old = json.loads(previous[0])
            lifecycle_fields = {"status", "valid_to", "superseded_by", "conflicts_with"}
            if {k: v for k, v in old.items() if k not in lifecycle_fields} != {
                k: v for k, v in fact.items() if k not in lifecycle_fields
            }:
                raise FactError("claim payloads are immutable")
        elif self._id_reserved(db, fact["fact_id"]):
            raise FactError("fact id permanently reserved")
        db.execute(
            "INSERT INTO facts VALUES(?,?,?,?,?) ON CONFLICT(fact_id) DO UPDATE SET payload=excluded.payload",
            (fact["fact_id"], *slot(fact), json.dumps(fact)),
        )
        self._audit(db, fact["fact_id"], action, prior, fact["status"])

    def ingest_candidate(self, fact):
        """Internal extraction hook; deliberately not an HTTP write endpoint."""
        self._enabled()
        validate("Fact", fact)
        semantics(fact)
        for provenance in [fact["provenance"], *fact.get("evidence", [])]:
            if provenance["evidence_date"] is not None:
                instant(provenance["evidence_date"])
        screen_retained_fact(fact)
        if fact["status"] != "candidate" or fact["source_class"] == "owner_stated":
            raise FactError("ingestion cannot assert owner authority")
        with self._db() as db:
            if self._id_reserved(db, fact["fact_id"]):
                raise FactError("fact id permanently reserved")
            key = self._suppression_key(db, fact)
            if any(
                json.loads(row[0]).get("suppression_key") == key
                for row in db.execute("SELECT payload FROM tombstones")
            ):
                raise FactError("source suppressed by retraction")
            self._save(db, copy.deepcopy(fact), "ingest")

    def draft(self, payload):
        self._enabled()
        self._prune_drafts()
        if not isinstance(payload, dict):
            raise FactError("invalid draft")
        draft = copy.deepcopy(payload)
        if any(k in draft for k in ("draft_id", "expires_at", "will_supersede")):
            raise FactError("server-owned draft fields")
        screen(draft)
        draft.update(
            draft_id="d_" + secrets.token_hex(12),
            expires_at=stamp(self.clock() + timedelta(hours=24)),
            will_supersede=[],
        )
        validate("CaptureDraft", draft)
        semantics(draft)
        probe = {
            k: copy.deepcopy(draft[k])
            for k in (
                "subject",
                "predicate",
                "value",
                "value_type",
                "unit",
                "privacy_class",
                "valid_from",
            )
        }
        probe.update(
            fact_id="f_00000000",
            status="candidate",
            source_class="conversation",
            provenance={"kind": "session", "ref": "pending", "evidence_date": None},
            confidence=0.7,
            recorded_at=stamp(self.clock()),
            last_verified=None,
            valid_to=draft.get("valid_to"),
            item_key=draft.get("item_key"),
        )
        validate("Fact", probe)
        origin = draft["origin"]["session_ref"]
        if not origin.strip() or "#" not in origin:
            raise FactError("origin must identify session#message")
        temporal = REGISTRY[draft["predicate"]][0]
        targets = draft.get("target_fact_ids", [])
        if draft["action"] in {"add", "replace"} and targets:
            raise FactError("targets are server-computed for replacements")
        if draft["action"] == "replace" and (
            temporal not in {"state", "set"} or not draft["valid_from"]
        ):
            raise FactError("replacement requires mutable state and known effective date")
        if draft["action"] in {"end", "retract", "replace"}:
            if not Path(self.path).exists():
                raise FactError("no target facts")
            with self._db() as db:
                rows = self._rows(db, *slot(draft))
            active = [
                f
                for f in rows
                if f["status"] != "retracted"
                and (
                    draft["action"] == "retract"
                    or (f["status"] != "superseded" and f["valid_to"] is None)
                )
            ]
            if draft["action"] == "replace":
                targets = [f["fact_id"] for f in active if f["source_class"] == "owner_stated"]
                draft["will_supersede"] = targets
                if not targets:
                    raise FactError("replacement requires existing facts")
            if (
                len(targets) != len(set(targets))
                or not targets
                or not set(targets) <= {f["fact_id"] for f in active}
            ):
                raise FactError("invalid target facts")
            if draft["action"] in {"end", "retract"} and any(
                f["value"] != draft["value"] for f in active if f["fact_id"] in targets
            ):
                raise FactError("read-back must match every target value")
            if draft["action"] == "end" and temporal not in {"state", "set"}:
                raise FactError("only state/set facts can end")
            boundary = (
                draft["valid_from"] if draft["action"] == "replace" else draft.get("valid_to")
            )
            for f in active:
                if (
                    f["fact_id"] in targets
                    and draft["action"] in {"replace", "end"}
                    and f["valid_from"]
                    and instant(boundary) <= instant(f["valid_from"])
                ):
                    raise FactError("transition must follow target start")
        with self._lock:
            if len(self._drafts) >= 100:
                raise FactError("pending draft limit")
            self._drafts[draft["draft_id"]] = draft
        return copy.deepcopy(draft)

    def confirm(self, draft_id, confirmation):
        self._enabled()
        self._prune_drafts()
        validate("Confirmation", confirmation)
        screen(confirmation)
        confirmation = {**confirmation, "assurance": "agent_asserted"}
        with self._lock:
            draft = self._drafts.get(draft_id)
            if not draft or instant(draft["expires_at"]) <= self.clock():
                self._drafts.pop(draft_id, None)
                raise FactError("draft missing or expired")
            origin_session, origin_msg = draft["origin"]["session_ref"].rsplit("#", 1)
            ref = confirmation["session_ref"]
            if (
                "#" not in ref
                or ref.rsplit("#", 1)[0] != origin_session
                or not ref.rsplit("#", 1)[1]
                or ref.rsplit("#", 1)[1] == origin_msg
            ):
                raise FactError("confirmation requires a distinct message in the draft session")
            if (
                not instant(draft["expires_at"]) - timedelta(hours=24)
                <= instant(confirmation["at"])
                <= self.clock()
            ):
                raise FactError("invalid confirmation time")
            created = []
            with self._db() as db:
                rows = self._rows(db, *slot(draft))
                targets = (
                    draft["will_supersede"]
                    if draft["action"] == "replace"
                    else draft.get("target_fact_ids", [])
                )
                active = [
                    f
                    for f in rows
                    if f["status"] != "retracted"
                    and (
                        draft["action"] == "retract"
                        or (f["status"] != "superseded" and f["valid_to"] is None)
                    )
                ]
                if targets and not set(targets) <= {f["fact_id"] for f in active}:
                    raise FactError("target changed; create a fresh draft")
                if draft["action"] == "replace" and set(targets) != {
                    f["fact_id"] for f in active if f["source_class"] == "owner_stated"
                }:
                    raise FactError("replacement changed; create a fresh draft")
                new_id = "f_" + secrets.token_hex(12)
                if draft["action"] in {"add", "replace"}:
                    conflicts = []
                    if draft["action"] == "add":
                        for old in active:
                            if old["source_class"] != "owner_stated":
                                continue
                            if (
                                REGISTRY[draft["predicate"]][0] == "series"
                                and old["valid_from"] != draft["valid_from"]
                            ):
                                continue
                            if old["value"] != draft["value"]:
                                conflicts.append(old)
                            else:
                                raise FactError(
                                    "same owner claim already exists; no duplicate current fact"
                                )
                    fact = {
                        k: copy.deepcopy(draft[k])
                        for k in (
                            "subject",
                            "predicate",
                            "value",
                            "value_type",
                            "unit",
                            "privacy_class",
                            "valid_from",
                        )
                    }
                    fact.update(
                        fact_id=new_id,
                        item_key=draft.get("item_key"),
                        valid_to=draft.get("valid_to"),
                        recorded_at=stamp(self.clock()),
                        last_verified=confirmation["at"],
                        status="current",
                        source_class="owner_stated",
                        confidence=0.95,
                        confirmation=copy.deepcopy(confirmation),
                        provenance={
                            "kind": "session",
                            "ref": ref,
                            "evidence_date": confirmation["at"],
                        },
                    )
                    if targets:
                        fact["supersedes"] = targets
                    if conflicts:
                        fact.update(
                            status="conflict", conflicts_with=[f["fact_id"] for f in conflicts]
                        )
                        for old in conflicts:
                            prior = old["status"]
                            old.update(
                                status="conflict",
                                conflicts_with=sorted(
                                    set(old.get("conflicts_with", [])) | {new_id}
                                ),
                            )
                            self._save(db, old, "conflict", prior)
                    self._save(db, fact, "confirm")
                    created.append(fact)
                for old in active:
                    if old["fact_id"] not in targets:
                        continue
                    prior = old["status"]
                    if draft["action"] == "retract":
                        # Purge value from current storage; retain only a minimal tombstone.
                        tombstone = dict(
                            fact_id=old["fact_id"],
                            reason="owner_forget",
                            at=stamp(self.clock()),
                            suppression_key=self._suppression_key(db, old),
                        )
                        validate("RetractionTombstone", tombstone)
                        db.execute("DELETE FROM facts WHERE fact_id=?", (old["fact_id"],))
                        db.execute(
                            "INSERT INTO tombstones VALUES(?,?)",
                            (old["fact_id"], json.dumps(tombstone)),
                        )
                        self._audit(db, old["fact_id"], "retract", prior, "retracted")
                    else:
                        old["valid_to"] = (
                            draft["valid_from"]
                            if draft["action"] == "replace"
                            else draft["valid_to"]
                        )
                        if draft["action"] == "replace":
                            old.update(status="superseded", superseded_by=new_id)
                        self._save(db, old, draft["action"], prior)
                if draft["action"] == "retract":
                    for survivor in self._rows(db, *slot(draft)):
                        if survivor["status"] == "conflict":
                            peers = [
                                f
                                for f in self._rows(db, *slot(draft))
                                if f["fact_id"] in survivor["conflicts_with"]
                                and f["status"] not in {"retracted", "superseded"}
                            ]
                            survivor["conflicts_with"] = [f["fact_id"] for f in peers]
                            if not peers:
                                survivor.pop("conflicts_with")
                                survivor["status"] = (
                                    "current"
                                    if survivor["source_class"] == "owner_stated"
                                    else "candidate"
                                )
                            self._save(db, survivor, "conflict_reconcile", "conflict")
                # Remove any other pending read-back retaining a forgotten value.
                if draft["action"] == "retract":
                    self._drafts = {k: v for k, v in self._drafts.items() if slot(v) != slot(draft)}
            self._drafts.pop(draft_id, None)
            return {
                "draft_id": draft_id,
                "action": draft["action"],
                "facts": created,
                "target_fact_ids": targets,
            }


def token_from_env(environ):
    token = environ.get("MEMORY_FACTS_WRITE_TOKEN", "").strip()
    path = environ.get("MEMORY_FACTS_WRITE_TOKEN_FILE")
    if path:
        try:
            token = Path(path).read_text().strip()
        except OSError:
            token = ""
    return token


def authorized(header, token):
    if not token or not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(
        hashlib.sha256(header[7:].encode()).digest(), hashlib.sha256(token.encode()).digest()
    )
