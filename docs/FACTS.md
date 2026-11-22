# Opt-in private fact capture

This lane is **disabled by default**. Existing index/search endpoints and their
network-isolation authentication assumptions do not change. No live fact store,
importer, automatic extraction, profile injection, or deployment is enabled by
this implementation.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMORY_FACTS_ENABLED` | `false` | Enable new fact routes (`true`, `yes`, `1`) |
| `MEMORY_FACTS_SQLITE_PATH` | `/data/facts.db` | Separate private SQLite file; configured overrides must be absolute, never an index DB |
| `MEMORY_FACTS_WRITE_TOKEN` | unset | Bearer secret for **all** fact routes, including reads |
| `MEMORY_FACTS_WRITE_TOKEN_FILE` | unset | Secret-file alternative; takes precedence; unreadable/empty fails closed |
| `MEMORY_FACTS_RECENCY_DAYS` | `60` | Conversation recency window |
| `MEMORY_FACTS_TIMEZONE` | `UTC` | Owner calendar timezone for derived ages |

No DB file is created on import, disabled operations, draft creation, or an unused
lane's read. The first enabled confirmation/ingest creates a mode-0600 file. Use
a private directory/volume and one DB/token per OpenClaw instance/principal. A
restart loses pending drafts; it does not promote them. Do not share the DB,
network, or bearer secret between principals. The token is sampled at startup;
rotating its file requires a service restart through the deployment's normal gate.
No token values are logged or returned. The bearer scheme is case-sensitive
(`Bearer`). Token length, pre-existing file modes and rate limiting remain
operator responsibilities on the isolated network; creation mode is not a
permission audit of an existing DB or token file.

## Trust split (important)

The authenticated principal is the **OpenClaw instance**, not a human owner. The
service checks a constant-time bearer comparison before parsing private writes
or retrieving facts. Missing/wrong credentials return 401. If enabled without a
configured readable nonempty token, every fact read/write returns 503 and startup
logs a warning. Existing routes retain their existing authentication behavior.
TLS/network isolation is still necessary; do not expose the internal HTTP service
to the internet.

The trusted agent must authenticate the human owner and establish an owner-private
session using transport metadata, not payload claims. It must bind the owner's
unambiguous approval to the exact displayed unexpired draft, perform semantic
secret screening, and send only that operation for confirmation. The service
cannot authenticate a person's `yes`, inspect chat reply metadata, verify source
truth, or establish an audience from `origin.audience` or `confirmation.by`.
**Those fields and `confirmation.session_ref` are supplied provenance, not proof
of human identity.** A compromised instance token can assert false owner facts.
Persisted owner assertions include `confirmation.assurance: "agent_asserted"`;
this field is required on stored owner facts (the service supplies it on confirm).
Resolved rank zero is named `T0_owner_asserted`, not a claim of a human-verified
curated file. Consumers must preserve this caveat along with provenance.
Group/unknown/other-principal turns must never call this lane. Restricted facts
are explicit-request-only, not automatic profile material. No automatic profile
injection is implemented here.

## API

All routes below exist only while the flag is enabled. Include
`Authorization: Bearer <configured secret>`. Responses and source pointers are
private. Never forward them into a group or another principal's context.

### Draft then confirm

`POST /facts/drafts` accepts a draft without server-owned `draft_id`, `expires_at`
or `will_supersede`. Synthetic example:

```json
{
  "action": "add",
  "subject": "person:owner",
  "predicate": "employment.employer",
  "value": "Example Organization",
  "value_type": "string",
  "unit": null,
  "valid_from": null,
  "privacy_class": "employment",
  "origin": {
    "session_ref": "session:example#proposal-message",
    "audience": "owner_private"
  }
}
```

Returns a `CaptureDraft` with an opaque ID, 24-hour expiry and computed target
list. Drafts are volatile, limited to 100 pending per process, never stored in
SQLite. Expired values are pruned on draft creation, reads and confirmation
attempts (including malformed confirmations), not just on a new draft. This is
operation-triggered pruning, not an idle background timer; restart drops all
pending values. Invalid/secret-like drafts fail before storage. Unknown/custom predicates
are blocked pending registry review; derived ages and volatile health snapshots
cannot be stored. Registry types, units, minimum privacy, set item keys, known
observation dates and exclusive interval ordering are checked. Dates use exactly
`YYYY-MM-DD`; timestamps use RFC3339 calendar date, `T`, `HH:MM:SS`, optional
fraction and mandatory `Z`/numeric offset. Invalid calendar dates/times, compact
dates, ISO-week forms, timezone-naive times and leap seconds are rejected before
a draft can be retained. Timestamp format validation is a declared runtime
dependency; the JSON Schema library is imported only on validation, not on the
disabled server startup path. Provenance has
no excerpt field; nested `excerpt` fields are rejected as well.

The agent displays the exact value, privacy, effective date, action and target
changes, then waits for a draft-bound owner reply (`yes`/`ja`/`да`/`ok`/👍 only
when unambiguous). Generic acknowledgement of another topic is insufficient.
Editing creates a fresh read-back; no approval means no fact write.

`POST /facts/confirm`:

```json
{
  "draft_id": "d_0123456789abcdef",
  "confirmation": {
    "by": "owner",
    "at": "2026-11-22T12:00:00Z",
    "session_ref": "session:example#confirmation-message"
  }
}
```

The draft ID, a distinct nonempty confirmation message in the same session,
and a confirmation timestamp between draft creation and now are required.
Expired/replayed/changed-target confirmations fail. This is transport validation,
not verification of the human message. Confirmation creates an owner-stated
fact or applies an approved lifecycle transition atomically with value-free audit
events. Curated-file edits are always separate reviewed diffs; this API never
writes workspace memory files.

### Lifecycle actions

- `add`: new state assertion or dated series observation. Distinct set keys are
  independent; series observations do not supersede prior dates. Differing owner
  claims without an explicit transition become linked conflicts, not a recency
  vote. Duplicate owner claims are rejected rather than creating duplicate current
  records; an explicit refresh operation can be added separately. Existing
  mutable state/set assertions can be re-confirmed via an explicitly displayed
  same-value `replace` with a known effective date; series can add a new dated
  observation. This is not an invisible freshness refresh.
- `replace`: state or same set key only, with a known `valid_from`. Server computes
  `will_supersede` from open-ended owner assertions only; unreviewed document/mail
  claims are not turned into proven historical states. Confirmation closes the
  explicitly displayed owner intervals at the stated date
  and creates symmetric replacement links. Unknown dates never close history.
- `end`: existing state/set targets plus a non-null exclusive `valid_to`; no new
  positive claim. The target value in the read-back must match. Other keys survive.
- `retract`: targets and matching read-back value; no new positive claim. This
  conservative implementation uses forget semantics for every retraction: purge
  payload with SQLite secure-delete, retain a minimal value-free tombstone and
  keyed source/slot suppression hash, and remove pending same-slot drafts.
  Ended/superseded history can also be forgotten. Re-extraction of the suppressed
  source/slot is rejected; a new explicitly confirmed owner assertion is allowed.
  Forgetting a replacement never resurrects its superseded predecessor: its old
  exclusive end remains, and now-tombstoned link IDs may remain for audit/history.
  A new assertion needs its own proposal/confirmation.

`POST /facts/end` and `/facts/retract` are action-checked aliases for **draft
creation**, not bypasses for confirmation. They take the corresponding draft
payload with `target_fact_ids`. `/facts/drafts` accepts all four actions.
The service returns generic errors without echoing private payloads. Append-only
lifecycle rows record timestamp, instance actor, IDs, action and prior/new status;
claim values are immutable except physical purging on forget. These events are
not a complete belief-time replay API. Backups and external views/caches require
a separate forgetting policy before restoring or activating an adapter; deleting
the current file's value does not erase external copies.

### Current-state lookup and unified-search integration

`GET /facts/current?subject=person:owner&predicate=employment.employer`

Optional: `item_key`, `as_of` (ISO date/time), `include_history=true`.
Returns `ResolvedFact`: selected assertion, alternatives, optional historical
assertions, authority rank/tier and disclosures. `person.age` derives calendar age
from DOB, including March 1 for leap-day birthdays in non-leap years; it propagates
conflict, unreviewed and privacy metadata and is never stored.

Explicit opt-in integration uses `/unified/search` with `fact_subject`,
`fact_predicate`, optional `fact_item_key`, `fact_as_of`, `fact_include_history`.
It authenticates before fact lookup and returns a top-level `current_state` block
**before** `results`. Without explicit fact parameters, the legacy response is
unchanged, even with the lane enabled. With the flag off, all new routes return
404 and the existing unified API ignores these extra parameters as before.
A fact backend failure is `unavailable`/`partial_lane_unavailable`, not a false
`not_found`. No free-text intent classifier, entity linker, automatic evidence
chunk relabeller, always-on slice, connector, or agent trigger is activated here.

## Resolver contract

`owner_stated` > conversation within the configurable recency window > newest
dated documents/expired conversations > older or unknown-dated evidence > mail.
Ranking uses verified source dates, not file mtimes or ingestion times. Tied
newest documents and unknown-date credible peers disagreeing with dated claims
produce conflicts. Equal-authority contradictory owner facts remain conflicts;
confidence and deterministic ID sorting order presentation, never truth.
Series chooses latest observation first, then authority for same-instant ties.
Static same-source disagreements are not settled by recency.

A newer credible document contradicting an owner fact is returned as a conflict
while retaining the owner's assertion as the lead. Mail cannot overwrite or
refresh it; mail-only output is `unverified_only`. Candidate selections remain
stored candidates and carry `unreviewed`. Unknown source dates/validity and stale
claims are disclosed; ended intervals are exclusive. Reads are SQLite read-only
and never modify lifecycle/status or infer missing transition dates.

Extraction must establish that evidence actually asserts the predicate/subject.
A billing address is not automatically a residence assertion; a service mention
is not ownership evidence. This service cannot infer those distinctions from a
pointer, nor authenticate source dates, alias equivalence, or OCR subject linkage.
Secret-pattern checks apply NFKC normalization and remove Unicode format (`Cf`)
characters before matching; nested payloads, passphrases, secrets/tokens,
PIN/OTP/CVV labels, common localized credential words, long hex/Base32 and
credential formats are rejected. These are deliberately conservative checks:
ordinary phrases such as a company name containing "Credential" can be false
positives. They are not proof that arbitrary secrets or sensitive identifiers
can be detected; card/IBAN classification is not implemented here. Agent-side semantic screening remains required.

Tests contain only invented records and generic references. The packaged JSON
Schema is a separately scrubbed structural specification, not a private design
or migration artifact. No production imports or real evidence are test inputs.

## Implementation housekeeping and non-goals

DDL initialization is guarded per database file identity within a store instance;
normal subsequent writes do not repeat `executescript`. A new store instance
checks/initializes its DB idempotently. The package version stays at the current
release number until the coordinated release process; this development branch
does not publish a release.

`ingest_candidate` remains an internal, synthetic-tested integration hook with no
production caller in this change. Its tombstone suppression lookup currently
scans rows (O(n)); indexing is deferred until a production connector provides a
measured workload. The HMAC salt is stored in the same private DB: suppression
keys avoid raw source text in tombstones but **are not anonymization against a DB
reader**. Low-entropy source/slot guesses can be tested by someone with that DB.
The isolation/token boundary, not that hash, protects private data.

## Retained-input and log boundaries

Credential screening runs before a confirmation opens any transaction and before
candidate ingestion creates/opens the DB. It covers every retained free-text
field, including notes, item keys, source/evidence refs, locators, extractor names
and confirmation message refs; lifecycle saves enforce the same invariant.
Only the schema-validated `content_sha256` fields directly inside provenance and
evidence provenance are exempt from credential heuristics. A similarly named key
inside an arbitrary structured claim is not an exemption.

Fact IDs are permanently reserved across live facts, tombstones and lifecycle
history. A forgotten ID cannot be ingested again even with a different source.
New assertions need fresh IDs; this preserves identity and allows subsequent
forgetting without overwriting prior tombstones.

Private-request access logging classifies decoded query keys and normalized
paths, including percent-encoded fact keys, errors and unauthorized requests.
It logs only an allowlisted route path (or a generic private path for unknown
fact routes), never the query string. Application query logging uses the same
classifier, including fact-marked requests that remain on the legacy search lane.
