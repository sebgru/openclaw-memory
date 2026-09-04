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

Promotion is intentionally a human decision. The search service never edits authoritative Markdown.

## Session archive

Session archives are append-only historical evidence, not automatically trusted memory. Configure `ARCHIVE_ROOT` to enable a separate SQLite database and, when Qdrant is enabled, a separate vector collection.

- `POST /archive/index` incrementally indexes archive Markdown.
- `GET /archive/search?q=...` searches only the archive.
- `GET /archive/status` reports archive counts.

Normal `/search` never falls through to the archive implicitly. Applications may call archive search explicitly or only when normal recall returns no useful result.
