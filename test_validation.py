import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import (EventPlanningOrchestrator, KnowledgeResearchAgent, OllamaError,
                    QualityAssuranceAgent, RequirementAnalysisAgent, SchedulePlanningAgent,
                    VenueLogisticsAgent, cosine_similarity)
from utils import budget_utils as bu

TODAY = date(2026, 1, 1)


def valid_input(**overrides):
    data = {"event_type": "Wedding", "event_date": date(2026, 3, 1), "location": "Pune",
            "guest_count": 100, "total_budget": 1_000_000, "venue_capacity": 150,
            "theme": "Royal pink", "food_preference": "Vegetarian", "special_requirements": ""}
    data.update(overrides)
    return data


GOOD_PLAN = (
    "## 1. Event Overview\nA wedding for 100 guests with a royal pink theme.\n"
    "## 2. Theme and Decoration\nUse draped fabric and flowers.\n"
    "## 3. Catering and Menu\nA vegetarian buffet with a dessert counter.\n"
    "## 4. Activities and Entertainment\nMusic and dance after dinner.\n"
    "## 5. Venue and Logistics\nArrange parking and service staff.\n"
    "## 6. Budget Notes\nThe allocations are estimates.\n"
    "## 7. Preparation Timeline\nStart planning early and compare written quotations.\n"
)

KB_TEXT = """# Test KB
## Wedding Packages
wedding wedding package wedding venue ceremony.
## Birthday Party Packages
birthday birthday cake games.
## Catering Options
catering catering menu buffet food.
## Budget Planning Guidelines
budget budget planning split.
"""


class FakeClient:
    """Stands in for Ollama: deterministic embeddings, canned plan text."""
    VOCAB = ["wedding", "birthday", "catering", "venue", "decoration", "budget", "timeline", "photography"]

    def __init__(self, fail_embed=False, fail_generate=False, plan=GOOD_PLAN):
        self.fail_embed, self.fail_generate, self.plan = fail_embed, fail_generate, plan

    def embed(self, texts):
        if self.fail_embed:
            raise OllamaError("embedding service down")
        return [[t.lower().count(w) + 0.01 for w in self.VOCAB] for t in texts]

    def generate(self, prompt, system=None, **kwargs):
        if self.fail_generate:
            raise OllamaError("llm down")
        return self.plan


@pytest.fixture
def kb_file(tmp_path):
    path = tmp_path / "kb.txt"
    path.write_text(KB_TEXT, encoding="utf-8")
    return path


# ---------------- Requirement Analysis ---------------- #
def test_valid_requirements_pass():
    res = RequirementAnalysisAgent().run(valid_input(), today=TODAY)
    assert res.status == "SUCCESS" and not res.errors
    assert res.data["requirements"]["guest_count"] == 100
    assert res.data["requirements"]["event_date"] == "2026-03-01"
    assert res.data["requirements"]["special_requirements"] == "None"


def test_missing_fields_reported():
    res = RequirementAnalysisAgent().run({"event_type": "Wedding"}, today=TODAY)
    assert res.status == "FAILED"
    assert "Missing required field" in res.errors[0]


@pytest.mark.parametrize("override", [
    {"guest_count": 0}, {"guest_count": -4}, {"guest_count": "many"}, {"guest_count": 10.5},
    {"venue_capacity": 0}, {"total_budget": 0}, {"total_budget": -100}, {"total_budget": "abc"},
    {"event_date": date(2025, 12, 1)}, {"event_date": "not-a-date"},
    {"event_type": "Funeral"}, {"food_preference": "Pizza only"}, {"location": " "},
])
def test_invalid_fields_rejected(override):
    res = RequirementAnalysisAgent().run(valid_input(**override), today=TODAY)
    assert res.status == "FAILED" and res.errors


def test_short_notice_is_warning_not_error():
    res = RequirementAnalysisAgent().run(valid_input(event_date=date(2026, 1, 5)), today=TODAY)
    assert res.status == "WARNING" and res.warnings


# ---------------- Venue capacity ---------------- #
def test_guests_exceeding_capacity_rejected():
    res = VenueLogisticsAgent().run({"guest_count": 300, "venue_capacity": 200, "event_date": "2026-03-01"})
    assert res.status == "FAILED" and res.data["capacity_ok"] is False
    assert res.data["suggestions"] and "exceeds" in res.errors[0]


def test_guests_equal_to_capacity_is_flagged_crowded():
    res = VenueLogisticsAgent().run({"guest_count": 200, "venue_capacity": 200, "event_date": "2026-03-01"})
    assert res.status == "WARNING" and res.data["capacity_ok"] is True


def test_comfortable_capacity_succeeds():
    res = VenueLogisticsAgent().run({"guest_count": 100, "venue_capacity": 200, "event_date": "2026-03-01"})
    assert res.status == "SUCCESS" and res.data["logistics"]["tables_needed"] == 10


# ---------------- Schedule ---------------- #
def test_schedule_is_chronological_and_has_tasks():
    req = RequirementAnalysisAgent().run(valid_input(), today=TODAY).data["requirements"]
    res = SchedulePlanningAgent().run(req, today=TODAY)
    offsets = [r["offset_minutes"] for r in res.data["timeline"]]
    assert offsets == sorted(offsets)
    assert res.data["preparation_tasks"]


# ---------------- Retrieval (RAG) ---------------- #
def test_cosine_similarity_basics():
    sims = cosine_similarity([1, 0], [[1, 0], [0, 1], [1, 1]])
    assert sims[0] == pytest.approx(1.0) and sims[1] == pytest.approx(0.0)


def test_retrieval_returns_most_relevant_section(kb_file):
    agent = KnowledgeResearchAgent(FakeClient(), kb_path=kb_file)
    top = agent.retrieve("wedding package", per_query=1, max_results=1)
    assert top[0]["title"] == "Wedding Packages" and top[0]["method"] == "semantic"


def test_missing_knowledge_base_handled(tmp_path):
    agent = KnowledgeResearchAgent(FakeClient(), kb_path=tmp_path / "missing.txt")
    res = agent.run({"event_type": "Wedding", "guest_count": 10, "total_budget": 50000,
                     "food_preference": "Vegetarian", "theme": "x", "special_requirements": "None"})
    assert res.status == "WARNING" and res.data["chunks"] == []


def test_empty_knowledge_base_handled(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")
    res = KnowledgeResearchAgent(FakeClient(), kb_path=path).run(
        {"event_type": "Wedding", "guest_count": 10, "total_budget": 50000,
         "food_preference": "Vegetarian", "theme": "x", "special_requirements": "None"})
    assert res.status == "WARNING" and "empty" in res.warnings[0].lower()


def test_embedding_failure_uses_keyword_fallback(kb_file):
    agent = KnowledgeResearchAgent(FakeClient(fail_embed=True), kb_path=kb_file)
    res = agent.run({"event_type": "Wedding", "guest_count": 10, "total_budget": 50000,
                     "food_preference": "Vegetarian", "theme": "wedding", "special_requirements": "None"})
    assert res.status == "WARNING" and res.data["method"] == "keyword fallback"
    assert res.data["chunks"]


# ---------------- Quality Assurance ---------------- #
def _full_inputs():
    req = RequirementAnalysisAgent().run(valid_input(), today=TODAY).data["requirements"]
    budget = bu.analyze_budget(req["total_budget"], req["guest_count"], req["event_type"], req["food_preference"])
    venue = VenueLogisticsAgent().run(req).data
    schedule = SchedulePlanningAgent().run(req, today=TODAY).data
    return req, budget, venue, schedule


def test_qa_passes_for_consistent_data():
    req, budget, venue, schedule = _full_inputs()
    res = QualityAssuranceAgent().run(req, {"chunks": [1]}, budget, venue, schedule, GOOD_PLAN * 2)
    assert res.data["result"] == "PASS"


def test_qa_fails_when_budget_does_not_sum():
    req, budget, venue, schedule = _full_inputs()
    budget["allocations"] = {c: 1 for c in bu.CATEGORIES}
    res = QualityAssuranceAgent().run(req, {"chunks": [1]}, budget, venue, schedule, GOOD_PLAN * 2)
    assert res.data["result"] == "FAIL"
    assert any("sum" in r.lower() for r in res.data["reasons"])


def test_qa_fails_when_guests_exceed_capacity():
    req, budget, venue, schedule = _full_inputs()
    req["guest_count"], req["venue_capacity"] = 300, 200
    res = QualityAssuranceAgent().run(req, {"chunks": [1]}, budget, venue, schedule, GOOD_PLAN * 2)
    assert res.data["result"] == "FAIL"


def test_qa_fails_when_plan_missing():
    req, budget, venue, schedule = _full_inputs()
    res = QualityAssuranceAgent().run(req, {"chunks": [1]}, budget, venue, schedule, "")
    assert res.data["result"] == "FAIL"


def test_qa_flags_fabricated_booking_claims_as_warning():
    req, budget, venue, schedule = _full_inputs()
    plan = GOOD_PLAN * 2 + "\nThe hall is confirmed booking for your date."
    res = QualityAssuranceAgent().run(req, {"chunks": [1]}, budget, venue, schedule, plan)
    assert res.data["result"] == "PASS" and res.status == "WARNING"


# ---------------- Orchestrator (end to end with mocks) ---------------- #
def _orchestrator(kb_file, **client_kwargs):
    return EventPlanningOrchestrator(client=FakeClient(**client_kwargs), kb_path=kb_file)


def _future_input(**overrides):
    return valid_input(event_date=date.today() + timedelta(days=60), **overrides)


def test_orchestrator_happy_path_is_validated(kb_file):
    result = _orchestrator(kb_file).run(_future_input())
    assert result["validation_status"] == "VALIDATED"
    assert [t["step"] for t in result["trace"]] == [1, 2, 3, 4, 5, 6, 7]
    assert result["qa"]["result"] == "PASS" and result["plan_source"] == "llm"


def test_orchestrator_flags_capacity_problem(kb_file):
    result = _orchestrator(kb_file).run(_future_input(guest_count=300, venue_capacity=200))
    assert result["validation_status"] == "NOT_VALIDATED"
    assert result["qa"]["result"] == "FAIL"
    assert result["plan_text"]  # a draft is still produced, but flagged


def test_orchestrator_flags_budget_overrun(kb_file):
    result = _orchestrator(kb_file).run(_future_input(total_budget=100000, guest_count=500, venue_capacity=600,
                                                       food_preference="Non-Vegetarian"))
    assert result["validation_status"] == "NOT_VALIDATED"
    assert result["budget"]["status"] == "OVERRUN"


def test_orchestrator_stops_on_invalid_input(kb_file):
    result = _orchestrator(kb_file).run(_future_input(total_budget=-5))
    assert result["validation_status"] == "INVALID_INPUT"
    assert result["plan_text"] == ""
    assert [t["status"] for t in result["trace"][1:]] == ["SKIPPED"] * 6


def test_orchestrator_uses_fallback_when_llm_fails(kb_file):
    result = _orchestrator(kb_file, fail_generate=True).run(_future_input())
    assert result["plan_source"] == "fallback"
    writer_step = [t for t in result["trace"] if t["step"] == 6][0]
    assert writer_step["status"] == "WARNING"
    assert result["plan_text"]


def test_orchestrator_survives_embedding_failure(kb_file):
    result = _orchestrator(kb_file, fail_embed=True).run(_future_input())
    assert result["knowledge"]["method"] == "keyword fallback"
    assert result["validation_status"] in ("VALIDATED", "NOT_VALIDATED")