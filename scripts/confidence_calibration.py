#!/usr/bin/env python3
"""Offline analysis of the low-confidence semantic floor against labelled results.

Input: JSON Lines, one record per returned search result, with fields
``semantic_similarity`` (float), ``relevant`` (bool, human/ground-truth label) and
optionally ``lexical_evidence`` / ``exact_match``. The service only applies
``LOW_CONFIDENCE_SEMANTIC_FLOOR`` to results without lexical/exact evidence, so by
default only those records are analysed. Read-only: it never talks to the service.

Usage: scripts/confidence_calibration.py records.jsonl [--floor 0.6] [--all-evidence]
"""

import argparse
import json
import math
import sys

Z95 = 1.959963984540054


def wilson_interval(successes, total, z=Z95):
    """Wilson score interval for a binomial proportion; (None, None) when total is 0."""
    if total == 0:
        return None, None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _bucket(records):
    relevant = sum(1 for r in records if r["relevant"])
    low, high = wilson_interval(relevant, len(records))
    return {
        "n": len(records),
        "relevant": relevant,
        "relevant_rate": relevant / len(records) if records else None,
        "ci95": [low, high],
    }


def calibrate(records, floor=0.6, lexical_free_only=True):
    """Split labelled results at ``floor`` and report relevance rates with 95% CIs."""
    if lexical_free_only:
        records = [
            r for r in records if not r.get("lexical_evidence") and not r.get("exact_match")
        ]
    below = [r for r in records if r["semantic_similarity"] < floor]
    at_or_above = [r for r in records if r["semantic_similarity"] >= floor]
    flagged_irrelevant = sum(1 for r in below if not r["relevant"])
    total_irrelevant = sum(1 for r in records if not r["relevant"])
    suppressed_relevant = sum(1 for r in below if r["relevant"])
    total_relevant = sum(1 for r in records if r["relevant"])
    return {
        "floor": floor,
        "n": len(records),
        "below_floor": _bucket(below),
        "at_or_above_floor": _bucket(at_or_above),
        # Share of irrelevant results the floor would flag, and of relevant ones it would flag.
        "irrelevant_flagged_rate": (
            flagged_irrelevant / total_irrelevant if total_irrelevant else None
        ),
        "relevant_flagged_rate": suppressed_relevant / total_relevant if total_relevant else None,
    }


def load_records(lines):
    return [json.loads(line) for line in lines if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", help="labelled results JSONL")
    parser.add_argument("--floor", type=float, default=0.6)
    parser.add_argument("--all-evidence", action="store_true", help="include lexical hits")
    args = parser.parse_args(argv)
    with open(args.records, encoding="utf-8") as handle:
        records = load_records(handle)
    report = calibrate(records, args.floor, lexical_free_only=not args.all_evidence)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
