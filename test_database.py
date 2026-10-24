import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import database as db


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_events.db"


def sample_event(**overrides):
    event = {"event_type": "Wedding", "event_date": "2026-12-01", "location": "Pune", "guest_count": 120,
             "total_budget": 800000, "venue_capacity": 200, "theme": "Royal", "food_preference": "Vegetarian",
             "special_requirements": "None"}
    event.update(overrides)
    return event


def test_database_file_created_automatically(db_path):
    assert not db_path.exists()
    db.init_db(db_path)
    assert db_path.exists()
    assert db.get_all_events(db_path) == []


def test_save_and_retrieve_event(db_path):
    event_id = db.save_event(sample_event(), "PLAN TEXT", "VALIDATED", "PASS", {"trace": [1, 2]}, db_path)
    event = db.get_event(event_id, db_path)
    assert event["event_type"] == "Wedding" and event["plan_text"] == "PLAN TEXT"
    assert event["result"] == {"trace": [1, 2]}
    assert event["validation_status"] == "VALIDATED" and "result_json" not in event


def test_get_all_events_newest_first(db_path):
    first = db.save_event(sample_event(), "a", "VALIDATED", "PASS", None, db_path)
    second = db.save_event(sample_event(location="Mumbai"), "b", "NOT_VALIDATED", "FAIL", None, db_path)
    events = db.get_all_events(db_path)
    assert [e["id"] for e in events] == [second, first]
    assert events[0]["guest_count"] == 120 and events[0]["total_budget"] == 800000


def test_missing_event_returns_none(db_path):
    assert db.get_event(999, db_path) is None


def test_delete_event(db_path):
    event_id = db.save_event(sample_event(), "x", "VALIDATED", "PASS", None, db_path)
    assert db.delete_event(event_id, db_path) is True
    assert db.get_event(event_id, db_path) is None
    assert db.delete_event(event_id, db_path) is False


def test_checklist_save_update_and_summary(db_path):
    event_id = db.save_event(sample_event(), "x", "VALIDATED", "PASS", None, db_path)
    count = db.save_checklist(event_id, [{"task": "Book caterer", "due_date": "2026-10-01"}, "Send invitations"], db_path)
    assert count == 2
    items = db.get_checklist(event_id, db_path)
    assert items[0]["completed"] is False
    assert db.update_checklist_item(items[0]["id"], True, db_path) is True
    summary = db.get_checklist_summary(event_id, db_path)
    assert summary == {"total": 2, "completed": 1, "remaining": 1}
    assert db.update_checklist_item(items[0]["id"], False, db_path) is True
    assert db.get_checklist_summary(event_id, db_path)["completed"] == 0


def test_add_checklist_item_and_replace(db_path):
    event_id = db.save_event(sample_event(), "x", "VALIDATED", "PASS", None, db_path)
    db.add_checklist_item(event_id, "Custom task", "2026-11-01", db_path)
    assert len(db.get_checklist(event_id, db_path)) == 1
    db.save_checklist(event_id, ["Only task"], db_path)
    assert [i["task"] for i in db.get_checklist(event_id, db_path)] == ["Only task"]


def test_deleting_event_removes_its_checklist(db_path):
    event_id = db.save_event(sample_event(), "x", "VALIDATED", "PASS", None, db_path)
    db.save_checklist(event_id, ["Task"], db_path)
    db.delete_event(event_id, db_path)
    assert db.get_checklist(event_id, db_path) == []


def test_parameterized_queries_store_hostile_text_literally(db_path):
    nasty = "Pune'); DROP TABLE events;--"
    event_id = db.save_event(sample_event(location=nasty), "x", "VALIDATED", "PASS", None, db_path)
    assert db.get_event(event_id, db_path)["location"] == nasty
    assert len(db.get_all_events(db_path)) == 1


def test_missing_required_value_raises_database_error(db_path):
    bad = sample_event()
    bad["event_type"] = None
    with pytest.raises(db.DatabaseError):
        db.save_event(bad, "x", "VALIDATED", "PASS", None, db_path)


def test_unopenable_database_raises_database_error(tmp_path):
    with pytest.raises(db.DatabaseError):
        db.init_db(tmp_path)  # a folder cannot be opened as a database file