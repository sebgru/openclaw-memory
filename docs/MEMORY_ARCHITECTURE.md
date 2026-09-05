# Tiered memory architecture

This service treats Markdown as the source of truth. Vector and full-text indexes are derived data and can always be rebuilt.

## Authoritative sources

A typical workspace uses this layout:

```text
MEMORY.md                    # small durable index and pointers
USER.md                      # durable user preferences and facts
memory/
├── YYYY-MM-DD.md            # recent daily observations
├── knowledge/               # reviewed, permanent knowledge
│   ├── decisions/
│   ├── people/
│   ├── projects/
│   ├── health/
│   └── infrastructure/
├── review-candidates/       # proposed facts awaiting approval
└── archive/
    └── sessions/YYYY/YYYY-MM/ # historical raw session records
```

Normal memory search should include `MEMORY.md`, `USER.md`, daily notes, and reviewed files below `memory/knowledge/`. It must exclude `memory/review-candidates/` because proposals are not facts. It also excludes `memory/archive/`; raw history is searched explicitly through the archive API.

Example configuration:

```text
DOCUMENT_ROOT=/data/workspace
INCLUDE_PATTERNS=MEMORY.md,USER.md,memory/**/*.md
EXCLUDE_PATTERNS=memory/review-candidates/**,memory/archive/**
ARCHIVE_ROOT=/data/workspace/memory/archive
```

## Promotion workflow

1. Capture recent observations in dated daily notes.
2. Extract durable candidates into `memory/review-candidates/`, retaining source and date metadata.
3. Review each candidate. Rejected items remain non-authoritative or are removed.
4. Promote approved facts or decisions to the appropriate `memory/knowledge/<category>/` file.
5. Run incremental indexing. Deterministic chunk identifiers replace only changed material.

The service exposes this workflow without making a judgment: `GET
/promotion/candidates` lists Markdown candidates, and `POST /promotion/promote`
requires the caller to name the candidate, a `memory/knowledge/*.md`
destination, and a non-empty `approved_by` value. Promotion is an explicit
human action, writes atomically, appends provenance to the promoted file, and
retains the candidate as an audit record. There is no background or automatic
promotion path.

Promotion is intentionally a human decision. Normal indexing and search never edit authoritative Markdown; only the explicit promotion action writes the selected destination.

## Session archive

Session archives are append-only historical evidence, not automatically trusted memory. Configure `ARCHIVE_ROOT` to enable a separate SQLite database and, when Qdrant is enabled, a separate vector collection.

- `POST /archive/index` incrementally indexes archive Markdown.
- `GET /archive/search?q=...` searches only the archive.
- `GET /archive/status` reports archive counts.

Archive maintenance is incremental: repeated `/archive/index` calls compare
SHA-256 digests, index only added/changed Markdown, and remove records for
deleted files. Archive search is always explicit; `/search` does not search or
fall through to the archive. The archive database and vector collection are
separate from the authoritative tier.

## Monitoring and recovery

`GET /healthz` runs SQLite's integrity check and returns `status: ok` plus
database counts; a failed check returns `status: error`. For independent
maintenance, run `python -m memory_store.maintenance verify --source
memory.db` or `python -m memory_store.maintenance backup --source memory.db
--destination backups/memory.db`. Backups use SQLite's online backup API and
verify both the backup and a temporary restore. Scheduling, retention, and
off-host storage remain deployment responsibilities; no Compose or service
configuration is assumed.

The repository includes `scripts/archive-maintenance.sh` for a deployment
scheduler. It uses a non-blocking `flock`, calls only `/archive/index`, and
returns exit code 75 when another run owns the lock. Schedule one instance per
archive endpoint; do not run overlapping requests or rebuild the native
OpenClaw index as part of this job.

Normal `/search` never falls through to the archive implicitly. Applications may call archive search explicitly or only when normal recall returns no useful result.

## Rebuild-loop monitoring

This service has no autonomous rebuild loop: indexing occurs only when a caller
invokes `POST /index` or `POST /archive/index`. The service therefore does not
schedule retries, mutate configuration, or run a background repair process.

An external scheduler may monitor each response's `added`, `changed`,
`removed`, and `unchanged` counters and treat a non-2xx response or a failed
`/healthz` integrity check as an alert. Retries should remain explicit and
single-owner; do not run overlapping index calls against the same database.
`GET /status` is an alias of `/healthz` for simple probes. This keeps rebuild
ownership and retention policy outside the service and avoids hidden rebuild
loops.

## Native OpenClaw index policy

The external adapter is the active integration boundary. It performs bounded
`GET /search` retrieval only and never invokes `/index` or archive search.
OpenClaw's bundled/native memory index remains a separate derived index and is
not to be rebuilt during rollout, archive population, or adapter outages.
Keep the adapter release and OpenClaw image pinned independently. A native
rebuild requires a separately reviewed migration plan, a verified backup, and
explicit operator approval; it is not an automatic recovery action.

## Acceptance sequence

After deployment, the operator should record `/healthz`, `/status`, and
`/archive/status`, run `/index` and `/archive/index` once, then test one known
authoritative query and one known archive query. Repeat both searches with the
embedding/Qdrant path unavailable to verify FTS fallback. Finally test the
adapter's WebChat and Telegram allowlists and an unavailable-service timeout.
Live results belong in the deployment handoff, not in this public repository.
