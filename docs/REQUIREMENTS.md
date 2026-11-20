# Requirements

Status: v1.1. These requirements define what this service is for and
what every change must preserve. They are derived from the deploying owner's
private system requirements and contain no personal data.

## Purpose

`openclaw-memory` is an external retrieval service for corpora that are too
large, too slow to rebuild, or too sensitive to live inside an OpenClaw
Gateway process:

- **Documents:** a read-only corpus of converted Markdown documents
  (household, family, finance, insurance, vehicles, contracts, health).
- **Long-term archive:** historical session records and older knowledge.

In the target architecture, OpenClaw's native memory (`memory-core`) owns
curated memory, daily notes, and recent sessions. This service complements it
and does not duplicate it. Overlapping main-memory scopes remain supported
for compatibility until the deploying owner retires them.

## Functional requirements

| ID | Priority | Requirement |
|---|---|---|
| SR-1 | Must | Index and search the documents corpus and the session archive as separate scopes with separate index state. Source files are never modified. |
| SR-2 | Must | Answer personal reference questions (for example "what is our address", "which insurance covers the car") from documents, returning the passage with source path, heading, line, and document date. |
| SR-3 | Must | Prefer current over historical evidence: expose document date (from filename, metadata, or content) so newer documents outrank older ones for current-state questions, and mark results that are likely superseded. |
| SR-4 | Must | Find earlier discussions and analyses in the archive by topic, returning the session identifier and date so the caller can open the original. |
| SR-5 | Must | Never let backup copies, trim snapshots, or duplicate files outrank the canonical file. Exclusion patterns and duplicate detection are explicit and testable. |
| SR-6 | Must | Cross-language retrieval: a query in one language (for example English) finds content written in another (for example German) with recall comparable to a same-language query, in every scope. Lexical and vector stages must both support this (for example query translation or expansion for lexical search); a cross-language query must not depend on the vector stage alone. |
| SR-7 | Should | Expose document category and conversion/OCR quality so callers and ranking can use them. |
| SR-8 | Should | Normalize scores across scopes so results from different scopes can be merged without one scope dominating by scale alone. |
| SR-9 | Could | Extract stable, dated facts (entity, attribute, value, source, date) from documents into a reviewable candidate list. Candidates are never promoted automatically. |
| SR-10 | Must | Return responses that a single client entry point can merge with results from other memory backends: per-result scope, provenance, normalized score, and date, so the client never needs a second search tool. |

## Non-functional requirements

| ID | Priority | Requirement |
|---|---|---|
| SN-1 | Must | Latency fits inside a 6 s client budget with headroom: target p50 ≤ 1 s and p95 ≤ 3 s per scoped search (proposed). |
| SN-2 | Must | Health and status endpoints are cheap and never run full integrity checks per call. They must not compete with search for resources. |
| SN-3 | Must | Report degraded paths explicitly: unavailable reranker, fallback ranking, partial scope coverage, low confidence, and timeouts appear in the response. |
| SN-4 | Must | Aborted client requests stop or bound their server-side work. |
| SN-5 | Must | Accept an optional request/correlation identifier and return per-stage timings (lexical, vector, rerank, total) per scope. |
| SN-6 | Must | Tenant isolation: one instance serves one owner's corpus; no cross-instance reads. |
| SN-7 | Must | Public-safe repository: no personal data, credentials, or deployment-specific paths in code, tests, fixtures, or docs. Use synthetic examples. |
| SN-8 | Should | Index freshness: documents and archive changes become searchable within a bounded, configurable delay; rebuilds never block search. |
| SN-9 | Should | Indexes remain derived data: a full rebuild from source files is always possible and never required for normal recovery. |

## Change log

- v1.1: SR-6 raised to Must and sharpened to cross-language retrieval. Added SR-10 (merge-ready responses for a single client entry point).

## Out of scope

- Converting source documents (a separate pipeline produces the Markdown).
- Writing to curated memory files or promoting facts.
- Replacing OpenClaw's native memory for curated memory, daily notes, or
  recent sessions.

## Changing these requirements

Requirement changes need the deploying owner's approval and are made in a
reviewed pull request. Pull requests cite the requirement IDs they serve and
state any requirement they weaken.
