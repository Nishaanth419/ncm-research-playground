"""SQLite persistence for experiments (stdlib sqlite3, JSON columns)."""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

INTERRUPTED = "interrupted: the server restarted before this run finished"

JSON_FIELDS = ("request", "outputs", "timing", "trace", "metrics", "events")
SUMMARY_FIELDS = ("id", "created_at", "finished_at", "model", "status", "error", "request", "outputs", "timing",
                  "metrics", "events")

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    request TEXT NOT NULL,
    outputs TEXT,
    timing TEXT,
    trace TEXT,
    metrics TEXT,
    events TEXT
);
CREATE INDEX IF NOT EXISTS experiments_created ON experiments(created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ExperimentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def _row(row: sqlite3.Row, fields: tuple[str, ...]) -> dict[str, Any]:
        return {k: (json.loads(row[k]) if k in JSON_FIELDS and row[k] is not None else row[k]) for k in fields}

    def create(self, eid: str, *, model: str, request: dict) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO experiments (id, created_at, model, status, request) VALUES (?, ?, ?, 'queued', ?)",
                      (eid, _now(), model, json.dumps(request)))

    def update(self, eid: str, **fields: Any) -> None:
        if fields.get("status") in ("completed", "failed"):
            fields.setdefault("finished_at", _now())
        cols = ", ".join(f"{k} = ?" for k in fields)
        vals = [json.dumps(v) if k in JSON_FIELDS else v for k, v in fields.items()]
        with self._lock, self._conn() as c:
            c.execute(f"UPDATE experiments SET {cols} WHERE id = ?", (*vals, eid))

    def get(self, eid: str) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(f"SELECT {', '.join(SUMMARY_FIELDS)} FROM experiments WHERE id = ?", (eid,)).fetchone()
        return self._row(row, SUMMARY_FIELDS) if row else None

    def get_field(self, eid: str, field: str) -> Any:
        assert field in JSON_FIELDS
        with self._conn() as c:
            row = c.execute(f"SELECT {field} FROM experiments WHERE id = ?", (eid,)).fetchone()
        return json.loads(row[0]) if row and row[0] is not None else None

    def recover_interrupted(self) -> list[str]:
        """Mark runs a previous server process left queued/running as failed.

        Their work queue lived in memory, so they can never finish. A final
        `generation.failed` event (flagged `recovered`) is appended so stream
        clients get a terminal event instead of waiting forever.
        """
        with self._conn() as c:
            rows = c.execute("SELECT id, events FROM experiments WHERE status IN ('queued', 'running')").fetchall()
        for row in rows:
            events = json.loads(row["events"]) if row["events"] else []
            events.append({"type": "generation.failed", "experiment_id": row["id"], "seq": len(events),
                           "t": events[-1]["t"] if events else 0.0, "error": INTERRUPTED,
                           "error_type": "Interrupted", "recovered": True})
            self.update(row["id"], status="failed", error=INTERRUPTED, events=events)
        return [r["id"] for r in rows]

    def list(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        fields = ("id", "created_at", "finished_at", "model", "status", "error", "request", "timing")
        with self._conn() as c:
            rows = c.execute(f"SELECT {', '.join(fields)} FROM experiments ORDER BY created_at DESC, rowid DESC "
                             "LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [self._row(r, fields) for r in rows]
