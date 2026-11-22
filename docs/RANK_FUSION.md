# Unified rank fusion

`UNIFIED_FUSION_MODE=rrf` replaces raw-score fusion for `/unified/search` only.
`UNIFIED_FUSION_MODE=legacy` is the default and rollback setting. No schema,
ingestion, model, index or deployment changes are required. Unknown values fail
explicitly. `/search`, `/archive/search` and `/documents/search` keep their
existing scoring contract.

## Algorithm

1. Retrieve lexical, exact, fuzzy and vector candidates in every selected scope.
   Identifier-like queries still run lexical **and** vector retrieval; there is
   no identifier-only routing shortcut. Vector failures retain lexical fallback,
   warnings and coverage/degraded fields.
2. Within each scope, combine lexical and semantic ranks using `1/(60+rank)`
   (one-based ranks), plus the existing bounded fuzzy rank contribution. Raw
   vector similarity is retained for confidence, not added to the ranking score.
   Exact/term-coverage promotion is converted to rank units (`bonus/61`), using
   the existing configurable lexical weights.
3. When Nimble is configured, interleave scope ranks for its candidate prefix
   and make one bounded reranking call across candidates;
   use its relative order **within** each scope only. It cannot bypass the final
   cross-scope prior. Its existing deadline, cap and fallback remain unchanged.
4. Collapse duplicate groups; combine distinct scope ranks by RRF. Multiple
   copies in one scope contribute only their best rank, not extra votes.
5. Add `(tier_prior + metadata_adjustment - copy_penalty)/60`. Stable IDs break
   ties deterministically. Apply existing prompt relevance gates/quotas or tool
   result limit after fusion.

| Current scope/source mapping | Prior |
| --- | ---: |
| Main memory (including curated files, knowledge/notes, projects) and outputs/artifacts | 4 |
| Main sessions; archive sessions dated within the last 30 days | 3 |
| Other/undated archive material | 2 |
| Documents | 1 |

This is a **soft prior**, not unconditional truth precedence or a conflict
resolver. Independent multi-scope evidence can outweigh a one-tier difference.
Main scope is not fine-grained enough to distinguish T0 facts from all notes;
outputs/projects retain its existing main mapping. Date recognition for archive
sessions uses optional `doc_date`, then `session_date`, then ISO dates from
existing transcript headings or provenance paths. Undated archive content stays
T2; filesystem modification time is never used. Future dates do not imply T1.

For documents, valid ISO `doc_date` adds `0.25/(1+age_days/365)` (future dates are
clamped to age zero). Numeric finite `ocr_quality` is clamped to [0,1] and adds
`-0.25*(1-quality)`. Fields may be top-level or in `metadata`; missing/invalid
values are neutral. Missing dates do not become fabricated dates. These are
bounded heuristics, not measured calibration or supersession labels.

## Canonicalization

Equal optional chunk `content_hash`, or equal case/whitespace-normalized chunk
text, forms a duplicate group across scopes. Normalized path copies also group
when heading and line coordinates match. Different chunks in a document are
not collapsed solely because their file path is equal. Relative document paths
are namespaced separately from workspace paths to avoid basename collisions.

Known backup/trim/copies directories, `.bak`/`.backup` suffixes, and
`-copy`/`-backup`/`-trim` or `(copy N)` stem suffixes are recognized. A retrieved
canonical path wins over its copies regardless of their original rank; canonical
text, confidence evidence and provenance are retained, with other provenance in
`alternate_provenance`. Unmatched copies receive a rank penalty of 1. A
`canonical_of` pointer also marks a non-canonical candidate. This does not read
or remove files, resolve a pointer to fetch absent canonical content, or compare
arbitrarily shifted chunks semantically. Query-time dedup cannot recover a
canonical candidate that retrieval never returned. Index hygiene remains needed.

## Confidence and debug privacy

Rank-prior scores are **not relevance probabilities**. `lexical_evidence`,
`semantic_similarity`, `exact_match` and `relevance_score` remain evidence fields;
low-confidence flagging and prompt thresholds use them, not fused scores.
Empty-result handling is unchanged. No-match is not a claim of nonexistence.

Add `debug=1` to `/unified/search` to include a per-result `fusion_debug` object:

```json
{
  "scope_ranks": {"main": 2, "archive": 1},
  "rrf_score": 0.03252247488101534,
  "tier_prior": 4.0,
  "metadata_adjustment": 0.0,
  "copy_penalty": 0.0,
  "duplicates_collapsed": 1,
  "post_fusion_rank": 1
}
```

Scope ranks are before cross-scope fusion (after any within-scope Nimble order),
and post-fusion rank is before prompt quota/limit filtering. Debug contains only
scope names, ranks/counts and numeric adjustments; it contains no query, snippet,
path or hash. It is returned to the caller, never logged. Results themselves
retain their normal private provenance. HTTP access logs omit query strings;
unified timing logs omit query text. Embedding retries/fallbacks and unified
backend fallback logs omit text and exception content.

## Rollout gate

The default remains **legacy**: existing consumers/tests assert unchanged legacy
numeric scores and Nimble's reorder-only result contract. Running the existing
suite with `UNIFIED_FUSION_MODE=rrf` exposes those three compatibility assertions.
The normal full suite passes with the new mode covered explicitly by synthetic
unit/API tests. Enabling a new default must wait for a deliberate score-contract
migration and a passing full suite under that default. Do not interpret opt-in
unit tests or a candidate-only offline replay as deployment approval or an
end-to-end relevance gain. No production configuration is changed by this work.
