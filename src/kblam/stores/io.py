"""Small IO primitives shared by the local store implementations."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any


def open_sqlite(path: Path) -> sqlite3.Connection:
    """Open a writable store database with the invariants used by both stores."""

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON metadata file without exposing a partially written file."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def unlink_existing(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
