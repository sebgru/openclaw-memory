#!/bin/sh
set -eu

# One owner at a time: archive indexing is incremental and must not overlap.
: "${MEMORY_ENDPOINT:?set MEMORY_ENDPOINT, e.g. http://memory-sebg:8080}"
LOCK_FILE="${ARCHIVE_MAINTENANCE_LOCK:-/tmp/openclaw-memory-archive.lock}"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "archive maintenance already running" >&2; exit 75; }

curl --fail --silent --show-error --max-time "${MEMORY_TIMEOUT_SECONDS:-120}" \
  -X POST "$MEMORY_ENDPOINT/archive/index"
printf '\n'
