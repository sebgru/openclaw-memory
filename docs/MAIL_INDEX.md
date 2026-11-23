# Private mail source (R18)

Default off. This change supplies an exporter, isolated index builder and a protected
`/unified/search?scope=mail` lane. It does **not** enable Graph access, scheduling or
mail recall in the adapter. The private-chat adapter integration remains a rollout gate.
Ordinary `all`, `main`, `archive`, and `documents` requests never search mail.

## Account-specific adapter (outside version control)

`scripts/export_mail.mjs /private/adapter.mjs /private/export-root` requires a module
exporting `get(url)` and `policy = {version, select(records)}`. `get` must issue only
Graph GET requests, use `Prefer: outlook.body-content-type="text", IdType="ImmutableId",
odata.maxpagesize=100`, refresh credentials privately, and honor throttling. Never
include credentials, actual account allow-lists, mail subjects or addresses in this
repository. `select` returns a Set of Graph IDs from the supplied full current snapshot.
It can apply an externally reviewed policy including sender aliases and a first-2-KiB
body check. Changing `policy.version` forces a full resync. Folder denies run before
policy: Archive and Deleted Items, including descendants, are never requested.
The folder names currently follow the reviewed English-name policy; validate localized
or renamed folder identities before using another mailbox.

Initial per-folder delta queries deliberately omit `$top`: Graph treats `$top` as a
total change limit on delta, not merely page size. Follow every nextLink until deltaLink.
Cursors, current metadata and selected original bodies are retained in private state;
excluded bodies are discarded. Retaining selected original bodies makes body-prefix
selection repeatable after quote removal in the indexed projection. Recipients and
list markers remain private policy state, never markdown. No attachment API is called.

## State and reconciliation

A completed export publishes `current` atomically to an immutable generation containing
`mail/*.md`, `state.json`, `stats.json`. Files are mode 0600, generation/root mode 0700.
One exporter runs at a time (`.sync.lock`, exclusive create). Errors leave the previous
snapshot and delta cursors untouched. Old generations are deleted only after publication.
Do not run indexing and export concurrently: the index builder takes the same lock.
After a crash, verify no process is active before manually removing a stale lock.
Expired delta cursors fail closed; change policy version to rebuild from Graph.

Graph ID and conversation ID are private state, not searchable text. Files use hashed
internetMessageId (fallback: normalized subject, sender, date and projected body).
Duplicate message copies map to one file. Different replies are retained; a whole thread
is never collapsed to its newest message. Standard explicit reply quotations are trimmed
only for reply subjects; forwarded letters remain intact. Ambiguous quote formats are
kept, favoring recall over aggressive deletion. Per-message results are collapsed after
chunk retrieval. Attachment **contents and names are not exported**, so no duplicate
attachment entry is introduced alongside the existing document source. Cross-corpus
near-duplicate body text is not claimed to be solved by this change.

## Build and service

Use an explicit absolute database path, distinct from existing scopes:

```sh
PYTHONPATH=src python -m memory_store.mail_index \
  --export-root /private/export-root --database /private/mail.sqlite \
  --dimensions 1024 --qdrant-url http://isolated-qdrant:6333 --collection mail-private
```

Use the existing embedding environment (`EMBEDDING_URL`, `EMBEDDING_MODEL`); use the
same model/dimension for build and search. Omit Qdrant only for a lexical smoke test;
such a run cannot establish bilingual semantic acceptance. Use a new SQLite database
when adding vectors or changing embedding model; unchanged files are not re-embedded.
No live service or collection is an appropriate dry-run destination.

After explicit rollout approval, configure `MAIL_SQLITE_PATH`, `MAIL_ACCESS_TOKEN` and
optionally `MAIL_QDRANT_COLLECTION` (must start `mail-`), plus the existing Qdrant URL.
The token must differ from other capabilities and must be held only by a trusted private
owner-chat adapter. The adapter must verify channel, owner identity and private-chat
context before sending `X-Mail-Access`; this token is not itself chat authentication.
Never expose it to browsers, model-generated tools, group chats or other users. Missing
or wrong tokens return 403 without opening a mail search. Sender, date, subject and
hashed mail filename accompany results. No body or query is logged by these new paths.

Public/shared callers cannot opt into mail by setting scope alone. The separate mail
lane must be merged into the single user-facing unified search by the adapter only
following its private-chat check. Do not enable this before that adapter work is reviewed.

## Validation

`node --test tests/test_export_mail.mjs`, `pytest`, `ruff check .`, `mypy src`.
Fixtures use invented identities and reserved `.invalid` addresses only.
