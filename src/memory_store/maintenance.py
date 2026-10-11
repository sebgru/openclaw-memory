"""Setup-independent database maintenance helpers."""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def verify_database(path: str | Path, timeout: float = 5.0) -> dict[str, str]:
    """Run SQLite's integrity check and return a machine-readable result."""
    try:
        with sqlite3.connect(str(path), timeout=timeout) as db:
            result = db.execute("PRAGMA integrity_check").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        return {"status": "error", "integrity": str(exc)}
    return {"status": "ok" if result == "ok" else "error", "integrity": result}


def database_availability(path: str | Path, timeout: float = 2.0) -> dict[str, str]:
    """Check that a database can be opened without scanning every page.

    Routine status polling must not run ``integrity_check`` on multi-gigabyte
    stores. Full integrity verification remains available via ``verify``.
    """
    try:
        uri = Path(path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=timeout)) as db:
            db.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
    except sqlite3.DatabaseError as exc:
        return {"status": "error", "integrity": "not_checked", "error": str(exc)}
    return {
        "status": "ok",
        "integrity": "not_checked",
        "check": "not_run",
        "note": "Full integrity verification is available via the verify command.",
    }


def backup_database(source: str | Path, destination: str | Path, verify_restore=True) -> dict:
    """Create a consistent SQLite backup and optionally verify a restore copy."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source)) as source_db, sqlite3.connect(str(destination)) as backup_db:
        source_db.backup(backup_db)
    result = {"backup": str(destination), "verification": verify_database(destination)}
    if verify_restore:
        restored = destination.with_suffix(destination.suffix + ".restore-check")
        try:
            with (
                sqlite3.connect(str(destination)) as backup_db,
                sqlite3.connect(str(restored)) as restore_db,
            ):
                backup_db.backup(restore_db)
            result["restore_verification"] = verify_database(restored)
        finally:
            restored.unlink(missing_ok=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify or back up a memory SQLite database")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--source", required=True)
    backup = sub.add_parser("backup")
    backup.add_argument("--source", required=True)
    backup.add_argument("--destination", required=True)
    args = parser.parse_args()
    output = (
        verify_database(args.source)
        if args.command == "verify"
        else backup_database(args.source, args.destination)
    )
    print(json.dumps(output, sort_keys=True))
