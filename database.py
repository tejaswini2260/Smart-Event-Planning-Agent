
import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "event_planner.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    """Create the database tables if they do not exist."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_date TEXT NOT NULL,
                location TEXT,
                guest_count INTEGER NOT NULL,
                budget REAL NOT NULL,
                venue_capacity INTEGER NOT NULL,
                theme TEXT,
                food_preference TEXT,
                special_requirements TEXT,
                generated_plan TEXT,
                checklist_json TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_event(details, generated_plan, checklist, event_id=None):
    """Insert a new event or update an existing event."""
    checklist_json = json.dumps(checklist, ensure_ascii=False)

    with get_connection() as connection:
        if event_id is None:
            cursor = connection.execute("""
                INSERT INTO events (
                    event_type, event_date, location, guest_count,
                    budget, venue_capacity, theme, food_preference,
                    special_requirements, generated_plan, checklist_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                details["event_type"],
                details["event_date"],
                details.get("location", ""),
                details["guest_count"],
                details["budget_in_rupees"],
                details["venue_capacity"],
                details.get("theme", ""),
                details.get("food_preference", ""),
                details.get("special_requirements", ""),
                generated_plan,
                checklist_json
            ))
            return cursor.lastrowid

        connection.execute("""
            UPDATE events SET
                event_type = ?,
                event_date = ?,
                location = ?,
                guest_count = ?,
                budget = ?,
                venue_capacity = ?,
                theme = ?,
                food_preference = ?,
                special_requirements = ?,
                generated_plan = ?,
                checklist_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            details["event_type"],
            details["event_date"],
            details.get("location", ""),
            details["guest_count"],
            details["budget_in_rupees"],
            details["venue_capacity"],
            details.get("theme", ""),
            details.get("food_preference", ""),
            details.get("special_requirements", ""),
            generated_plan,
            checklist_json,
            event_id
        ))

        return event_id


def get_all_events():
    """Return saved events, newest first."""
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT id, event_type, event_date, location,
                   guest_count, budget, created_at
            FROM events
            ORDER BY id DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_event(event_id):
    """Return one complete saved event."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM events WHERE id = ?",
            (event_id,)
        ).fetchone()

    if row is None:
        return None

    event = dict(row)
    event["checklist"] = json.loads(
        event.get("checklist_json") or "[]"
    )
    return event


def update_checklist(event_id, checklist):
    """Save checklist completion states."""
    with get_connection() as connection:
        connection.execute("""
            UPDATE events
            SET checklist_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            json.dumps(checklist, ensure_ascii=False),
            event_id
        ))


def delete_event(event_id):
    """Delete a saved event."""
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM events WHERE id = ?",
            (event_id,)
        )
