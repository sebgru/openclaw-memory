# Unified rank fusion

`UNIFIED_FUSION_MODE=rrf` selects rank-only scoring for `/unified/search`.
`legacy` remains the default and rollback. No ingestion/schema/configuration
changes are required. Invalid values produce a JSON HTTP 500 (library calls
raise `ValueError`), not a dropped connection. Every successful unified response
has `fusion: "legacy"|"rrf"`. In both modes `score` is an opaque **ordering key**,
not a relevance probability; never compare scores between modes or requests.
Other search endpoints retain their legacy numeric scoring contract.

## Formula and scope mapping

1. Every scope retrieves lexical/exact, fuzzy and vector candidates. Identifier
   queries still run both lexical and vector search. Semantic failures disclose
   lexical fallback in warnings/coverage/degraded fields.
2. Within a scope, sum one-based `1/(60+rank)` lexical/vector votes, the bounded
   fuzzy contribution, and existing exact/term promotion expressed as
   `bonus/(60+1)`. Vector similarity is retained only as confidence evidence.
3. Group duplicates **before** Nimble. Interleave canonical scope ranks for its
   tier-agnostic bounded prefix, skipping groups already selected. Make one
   bounded call. If its order is unchanged (including disabled/fallback), keep
   the original lane ranks exactly and reuse the first fusion result. Otherwise
   reorder moved groups only among their own members/slots in each lane, leaving
   unaffected members in their original order. Explicit member-to-group mapping
   includes transitive identity bridges. Original members remain
   available for final evidence/provenance aggregation.
4. For a group with best rank `r_s` in scope `s`, final score is:
   `sum_s 1/(60 + r_s + tier_offset - metadata_bonus + copy_offset)`.
   Copies in the same scope contribute only the best rank. Distinct scopes vote
   once each. Most corpora are disjoint: for these rows cross-scope RRF is
   positional interleaving, not additional independent semantic evidence.
5. Break ties by tier, best scope rank, path and ID. Compute low confidence over
   the **full fused list**, then apply prompt relevance gates/quotas and limit.

| Source (current mapping) | Tier | Rank offset |
| --- | --- | ---: |
| Main memory, knowledge/notes, projects, outputs/artifacts | T0 | 0 |
| Main sessions; archive dated within the past 30 days | T1 | 1 |
| Other/undated archive | T2 | 2 |
| Documents | documents | 4 |

Offsets are a deliberately small, uncalibrated **soft prior in rank units**.
With one scope vote each and neutral metadata, document rank 1 has the score of
memory rank 5: memory rank 3 wins, but document rank 1 beats memory rank 6 (and
10). The equal-score boundary favours the higher tier. Multiple independent
scope votes can exceed a tier difference. Main does not distinguish curated
facts from every other note; this is neither truth precedence nor supersession.
Archive date recognition uses optional `doc_date`, then `session_date`, then ISO
dates in existing headings/paths. Future dates are not recent sessions. UTC date
is computed once per fusion call and is injectable for deterministic tests.

### Optional metadata

Document dates between 1900-01-01 and today's UTC date add the rank bonus
`1/(1+age_days/365)-0.5`: one-year-old is neutral, newest approaches +0.5,
and old approaches -0.5. Future/implausible/invalid dates are missing, not new.
Finite `ocr_quality` in [0,1] adds
`clamp((quality-0.9)*5, -0.5, 0.5)`: measured quality 0.9 and unknown are both
neutral. Each field moves at most half a rank; combined magnitude is at most one
rank. This is intended as small document-within-tier adjustment, not a measured
quality calibration. Missing values do not fabricate dates or imply infinite
age; top-level fields and nested `metadata` are supported.

## Conservative canonicalization

An explicit equal chunk `content_hash` groups duplicates. Bare normalized text
alone groups only session/archive rows and only for at least 80 characters.
Identical boilerplate chunks in different documents or ordinary notes are not
merged just because the text matches. Copy-normalized path + heading + line must
also have **equal normalized text**; different versions such as `q3.md` and
`q3-trim.md` with different content remain distinct. Document and workspace
relative paths are separate namespaces. This intentionally misses some rewritten
or shifted copies rather than hiding distinct sources.

Known backup/trim/copies directories, `.bak`/`.backup`, `-copy`/`-backup`/`-trim`
and `(copy N)` patterns identify copy candidates, not sufficient merge evidence.
A retrieved canonical path wins over a copy; canonical text is retained, while
`lexical_evidence`, `semantic_similarity`, and `relevance_score` are max-merged
and `exact_match` is OR-ed across all members. Alternate provenance is retained.
An unmatched copy or `canonical_of` marker costs **3 rank positions**. Nothing
reads/modifies files or fetches an absent canonical source. Optional hashes are
assumed to identify chunks, not whole multi-chunk files.

## Debug and privacy

HTTP `debug=1`, or `unified_search_with_coverage(..., debug=True)`, returns
`fusion_debug` in RRF mode. Library and HTTP defaults strip it; legacy debug is
harmless and has no fusion object. Fields contain no query, snippets, paths or
hashes:

- `engine_ranks`: lexical/vector/fuzzy ranks from the retrieved group members,
  before Nimble (numeric ranks only).
- `scope_ranks`: scope ranks after Nimble and before cross-scope fusion.
- `legacy_candidate_rank`: pre-Nimble raw legacy-style order **over the same
  retrieved candidate pool**, not an end-to-end shadow legacy request. It cannot
  reveal candidates already pruned by RRF retrieval or reproduce legacy duplicate
  score summing. Null means the row lacks identity evidence to correlate it.
- `rrf_score`: unadjusted scope-vote sum; `tier_prior`: diagnostic tier label
  4/3/2/1; `rank_offset`: actual offset; `metadata_adjustment`, `copy_penalty`,
  `duplicates_collapsed`: component diagnostics.
- `post_fusion_rank`: before profile/limit; `post_profile_rank`: contiguous rank
  in the delivered list (the first may have gaps, the second does not).

Confidence uses evidence fields, never rank scores. In RRF it is captured before
profile filtering/truncation; legacy/default mode preserves main (pre-E.2)
semantics by evaluating only the delivered rows after those steps. Empty-result
behaviour is unchanged. The tuple library API is unchanged; the HTTP layer
consumes request-thread confidence only in RRF mode.

Privacy hardening is intentionally shared by legacy and RRF: HTTP access logs
omit query strings (malformed requests log only a generic rejection), and unified **and scope-specific** timing logs omit queries.
Embedding retries/fallback and unified backend failures omit exception content.
This does not promise redaction of arbitrary failures elsewhere in the service;
result provenance remains private response data, not log data.

## Rollout and evaluation gate

Both default and forced-RRF suites must pass. Legacy score-contract tests are
mode-aware: legacy numeric assertions remain, while RRF asserts order, evidence
and provenance instead of unchanged raw scores. Legacy is still the default.
Default promotion also needs downstream score-consumer audit and an owner-labelled
end-to-end comparison (recall@10, MRR, regressions, dedup precision, latency with
and without Nimble). A candidate-limited reconstruction from saved responses is
only diagnostic: it exercises actual within-scope code but cannot recover
missing engine candidates or original rank gaps, establish owner relevance, or
measure service/Nimble latency. It is not production activation approval.


## Known full-depth design limit

The tier offset is evidence-blind: it does not vanish for an exact/lexical
match. With 300 weak main neighbours, 300 recent-archive neighbours and a rank-1
exact document, that document reaches overall **rank 10**, not rank 1. A
full-depth synthetic regression records this current policy limitation, not a
relevance acceptance criterion. Exact-match promotion still applies *within*
the document lane; cross-scope rank fusion discards its absolute magnitude.
Possible evidence-aware offsets or label-driven retuning require a separate
ranking decision and evaluation before default promotion. Compressed candidate
replays cannot validate this full-depth behaviour.

## Full-depth implementation overhead

A synthetic in-process profile (3 lanes x 300 candidates, tool limit 100, mocked
retrieval, Nimble disabled, 25 calls per mode) measured RRF median **51.34 ms**
before the lane-rank fix/cache optimization versus **18.99 ms** afterwards;
legacy medians were **3.18 / 2.69 ms** respectively. This is still roughly 7x
legacy's fusion-path overhead, not a service latency measurement. It excludes
network, embedding and retrieval latency; enabled Nimble can require a second
fusion after an actual reorder. Identities are now computed once per original
row and reused, candidate-legacy debug sorting is debug-only, and unchanged
Nimble results reuse the first fused grouping. Profile scripts/results stay in
the private task outputs, with synthetic fixtures only. No production latency
claim or default-promotion approval follows.
