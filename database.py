"""
database.py - SQLite persistence for events and preparation checklists.

Tables:
  events           - one row per saved event plan (+ full orchestrator result as JSON)
  checklist_items  - preparation tasks linked to an event (deleted with the event)
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "event_planner.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    event_date TEXT NOT NULL,
    location TEXT NOT NULL,
    guest_count INTEGER NOT NULL,
    total_budget INTEGER NOT NULL,
    venue_capacity INTEGER NOT NULL,
    theme TEXT,
    food_preference TEXT,
    special_requirements TEXT,
    validation_status TEXT NOT NULL,
    qa_result TEXT,
    plan_text TEXT NOT NULL,
    result_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checklist_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    task TEXT NOT NULL,
    due_date TEXT,
    completed INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_checklist_event ON checklist_items(event_id);
"""


class DatabaseError(Exception):
    """Raised when a database operation fails (message is safe to show to users)."""


def _resolve(db_path: Optional[Union[str, Path]]) -> str:
    return str(db_path) if db_path else str(DEFAULT_DB_PATH)


@contextmanager
def get_connection(db_path: Optional[Union[str, Path]] = None) -> Iterator[sqlite3.Connection]:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(_resolve(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)  # creates tables if the file is new
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn is not None:
            conn.rollback()
        raise DatabaseError(f"Database error: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Create the database file and tables if they do not exist."""
    with get_connection(db_path):
        pass


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _date_text(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value)


def save_event(
    event: Dict[str, Any],
    plan_text: str,
    validation_status: str,
    qa_result: str,
    result: Optional[Dict[str, Any]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Insert an event + generated plan. Returns the new event ID."""
    result_json = json.dumps(result, default=str) if result is not None else None
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO events (event_type, event_date, location, guest_count, total_budget,
                   venue_capacity, theme, food_preference, special_requirements,
                   validation_status, qa_result, plan_text, result_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.get("event_type"),
                _date_text(event.get("event_date")),
                event.get("location"),
                event.get("guest_count"),
                event.get("total_budget"),
                event.get("venue_capacity"),
                event.get("theme"),
                event.get("food_preference"),
                event.get("special_requirements"),
                validation_status,
                qa_result,
                plan_text,
                result_json,
                _now(),
            ),
        )
        return int(cur.lastrowid)


def get_all_events(db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """All saved events (newest first), without the large plan text."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT id, event_type, event_date, location, guest_count, total_budget,
                      venue_capacity, theme, food_preference, validation_status, qa_result, created_at
               FROM events ORDER BY id DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_event(event_id: int, db_path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    """One event including plan_text and the parsed 'result' dict. None if not found."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if row is None:
        return None
    event = dict(row)
    raw = event.pop("result_json", None)
    try:
        event["result"] = json.loads(raw) if raw else None
    except ValueError:
        event["result"] = None
    return event


def delete_event(event_id: int, db_path: Optional[Union[str, Path]] = None) -> bool:
    """Delete an event (and its checklist). Returns True if a row was deleted."""
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        return cur.rowcount > 0


def save_checklist(
    event_id: int,
    tasks: List[Union[str, Dict[str, Any]]],
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Replace the event's checklist. Each task is a string or {'task':..., 'due_date':...}."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM checklist_items WHERE event_id = ?", (event_id,))
        count = 0
        for t in tasks:
            if isinstance(t, dict):
                text, due = str(t.get("task", "")).strip(), t.get("due_date")
            else:
                text, due = str(t).strip(), None
            if not text:
                continue
            conn.execute(
                "INSERT INTO checklist_items (event_id, task, due_date, completed, updated_at) VALUES (?, ?, ?, 0, ?)",
                (event_id, text, _date_text(due) if due else None, _now()),
            )
            count += 1
        return count


def add_checklist_item(
    event_id: int, task: str, due_date: Optional[Any] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO checklist_items (event_id, task, due_date, completed, updated_at) VALUES (?, ?, ?, 0, ?)",
            (event_id, task.strip(), _date_text(due_date) if due_date else None, _now()),
        )
        return int(cur.lastrowid)


def get_checklist(event_id: int, db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT id, event_id, task, due_date, completed, updated_at
               FROM checklist_items WHERE event_id = ?
               ORDER BY (due_date IS NULL), due_date, id""",
            (event_id,),
        ).fetchall()
    items = [dict(r) for r in rows]
    for item in items:
        item["completed"] = bool(item["completed"])
    return items


def update_checklist_item(item_id: int, completed: bool, db_path: Optional[Union[str, Path]] = None) -> bool:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "UPDATE checklist_items SET completed = ?, updated_at = ? WHERE id = ?",
            (1 if completed else 0, _now(), item_id),
        )
        return cur.rowcount > 0


def get_checklist_summary(event_id: int, db_path: Optional[Union[str, Path]] = None) -> Dict[str, int]:
    items = get_checklist(event_id, db_path)
    done = sum(1 for i in items if i["completed"])
    return {"total": len(items), "completed": done, "remaining": len(items) - done}