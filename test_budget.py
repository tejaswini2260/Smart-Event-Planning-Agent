import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import budget_utils as bu


@pytest.mark.parametrize("event_type", bu.EVENT_TYPES)
@pytest.mark.parametrize("total", [5000, 99999, 100001, 500000, 1234567, 7777777])
def test_allocations_sum_exactly_to_total(event_type, total):
    alloc = bu.allocate_budget(total, event_type)
    assert sum(alloc.values()) == total
    assert set(alloc) == set(bu.CATEGORIES)
    assert all(v >= 0 for v in alloc.values())


@pytest.mark.parametrize("event_type", bu.EVENT_TYPES)
def test_percentage_tables_sum_to_100(event_type):
    assert sum(bu.ALLOCATION_PERCENTAGES[event_type].values()) == 100


@pytest.mark.parametrize("bad", [0, -5, "abc", None, float("nan"), float("inf"), True, 4000])
def test_invalid_budgets_rejected(bad):
    ok, _, message = bu.validate_budget(bad)
    assert ok is False and message


def test_valid_budget_accepted():
    ok, amount, _ = bu.validate_budget(250000.0)
    assert ok and amount == 250000


def test_allocate_budget_rejects_non_integer():
    with pytest.raises(ValueError):
        bu.allocate_budget(1000.5, "Wedding")


def test_overrun_detected_and_suggestions_given():
    analysis = bu.analyze_budget(100000, 500, "Wedding", "Non-Vegetarian")
    assert analysis["status"] == "OVERRUN"
    assert analysis["catering_shortfall"] > 0
    assert analysis["suggestions"]


def test_no_overrun_for_comfortable_budget():
    analysis = bu.analyze_budget(1_000_000, 100, "Wedding", "Vegetarian")
    assert analysis["status"] == "OK" and analysis["catering_shortfall"] == 0
    assert analysis["sum_matches_total"] is True


def test_suggested_minimum_budget_resolves_the_overrun():
    first = bu.analyze_budget(100000, 500, "Wedding", "Non-Vegetarian")
    fixed = bu.analyze_budget(first["suggested_min_budget"], 500, "Wedding", "Non-Vegetarian")
    assert fixed["status"] == "OK"


def test_suggested_max_guests_resolves_the_overrun():
    first = bu.analyze_budget(100000, 500, "Wedding", "Non-Vegetarian")
    fixed = bu.analyze_budget(100000, first["suggested_max_guests"], "Wedding", "Non-Vegetarian")
    assert fixed["status"] == "OK"


@pytest.mark.parametrize("amount,expected", [(500, "₹500"), (100000, "₹1,00,000"), (1234567, "₹12,34,567"), (-2500, "-₹2,500")])
def test_format_inr(amount, expected):
    assert bu.format_inr(amount) == expected


def test_compare_spending_detects_overspend():
    alloc = bu.allocate_budget(100000, "Birthday")
    spending = dict(alloc)
    spending["Venue"] = alloc["Venue"] + 5000
    comp = bu.compare_spending(alloc, spending, 100000)
    assert comp["over_categories"] == ["Venue"]
    assert comp["overspent"] is True
    assert math.isclose(comp["remaining"], -5000)


def test_compare_spending_within_budget():
    alloc = bu.allocate_budget(100000, "Birthday")
    comp = bu.compare_spending(alloc, {k: v / 2 for k, v in alloc.items()}, 100000)
    assert comp["overspent"] is False and comp["over_categories"] == []