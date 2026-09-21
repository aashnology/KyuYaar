"""
The strength scale is defined once (src/strength.py) and every consumer reads
what the tools actually produce. These tests exist so that anything built on
the strength label, such as recommendation ranking, cannot be wired to a
different scale than the one the tools grade with.
"""

import re
from pathlib import Path

import pytest

import charts
from data_loader import load_data, scenario_dir
from decisions import build_options
from orchestrator import investigate
from strength import ACTIONABLE, BRIEF_NAME, RANK, STRENGTH_ORDER, WORD, is_actionable, weakest
from synthetic import SCENARIOS
from toolkit import Toolkit

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="module")
def investigations():
    out = {}
    for name in SCENARIOS:
        orders, marketing = load_data(scenario_dir(name))
        events = list(investigate("Revenue dropped last month. Why?", Toolkit(orders, marketing)))
        out[name] = events[-1].data["investigation"]
    return out


def test_every_table_describes_the_same_three_levels():
    levels = set(STRENGTH_ORDER)
    assert levels == {"weak", "moderate", "strong"}
    assert set(RANK) == set(WORD) == set(BRIEF_NAME) == set(charts.STRENGTH_COLOR) == levels


def test_the_brief_names_map_onto_the_code_labels():
    assert BRIEF_NAME == {"strong": "High", "moderate": "Medium", "weak": "Low"}
    assert ACTIONABLE == ("strong", "moderate")


@pytest.mark.parametrize("label,expected", [("strong", True), ("moderate", True), ("weak", False)])
def test_only_high_and_medium_evidence_is_actionable(label, expected):
    assert is_actionable(label) is expected


@pytest.mark.parametrize("wrong", ["High", "Medium", "Low", "STRONG", "", None])
def test_a_gate_fed_the_wrong_vocabulary_fails_loudly_instead_of_passing_silently(wrong):
    # The failure this guards against: logic written against the brief's High/Medium/Low
    # names quietly treating every real label ("strong", "weak") as not actionable.
    with pytest.raises(ValueError):
        is_actionable(wrong)


def test_weakest():
    assert weakest(["strong", "moderate"]) == "moderate"
    assert weakest(["strong", "weak", "moderate"]) == "weak"
    assert weakest([]) == "n/a"


def test_every_tool_emits_only_labels_on_the_scale(investigations):
    seen = set()
    for name, inv in investigations.items():
        assert inv.evidence, name
        for ev in inv.evidence:
            assert ev.strength in STRENGTH_ORDER, (name, ev.id, ev.strength)
            seen.add(ev.strength)
    assert seen == set(STRENGTH_ORDER)          # the scenarios exercise all three levels


def test_no_option_is_built_on_weak_evidence(investigations):
    checked = 0
    for name, inv in investigations.items():
        by_id = {e.id: e for e in inv.evidence}
        for option in build_options(inv.evidence).options:
            for evidence_id in option.addresses:
                assert evidence_id in by_id, (name, option.id, evidence_id)
                assert is_actionable(by_id[evidence_id]), (name, option.id, evidence_id)
                checked += 1
    assert checked > 0


def test_a_scenario_with_no_cause_offers_no_action_option(investigations):
    options = build_options(investigations["flat"].evidence)
    assert options.insufficient
    assert [o.kind for o in options.options] == ["hold"]


def test_no_module_keeps_a_private_copy_of_the_actionable_threshold():
    pattern = re.compile(r"""\(\s*["']strong["']\s*,\s*["']moderate["']\s*\)|_ACTIONABLE\s*=""")
    offenders = [p.name for p in SRC.glob("*.py") if p.name != "strength.py" and pattern.search(p.read_text())]
    assert offenders == []
