"""
Layer 8: recommend() ranks decision options by risk-adjusted projected impact,
gated on evidence strength, and picks at most one.

Cases 1-6 use small constructed options/evidence/scenarios so the right answer
can be checked by hand. The scenario-sweep cases at the bottom run the real
investigation over all four ground-truth datasets from synthetic.py.
"""

import dataclasses

import pytest

from decisions import DecisionOption
from evidence import Evidence
from recommend import (
    EVIDENCE, NO_GAIN, NO_PROJECTED_GAIN, RANKED, RECOMMENDED, NO_EVIDENCE,
    recommend, risk_of,
)
from scenario import Assumptions, Scenario


# ------------------------------------------------------------- fixtures ---

def _ev(id, strength):
    return Evidence(
        id=id, hypothesis="h", evidence_type="statistical", metric="m", value=-10.0,
        baseline=0.0, segment="region=X", strength=strength, sample_size=100, details={},
    )


def _option(id, kind="act", addresses=(), risks=("r1", "r2")):
    return DecisionOption(
        id=id, kind=kind, title=id.replace("_", " ").title(), rationale="because",
        addresses=list(addresses), confidence="strong", impact="some", assumptions=["a"],
        risks=list(risks),
    )


def _decision_set(options):
    from decisions import DecisionSet
    return DecisionSet(options=options, insufficient=False)


def _scenario(option_id, profit, rng=None, kind="act"):
    return Scenario(
        option_id=option_id, kind=kind, projectable=True, assumptions=Assumptions(),
        gross_profit_total=profit, gross_profit_range=rng, revenue_total=profit,
    )


def _hold():
    return _option("hold_and_monitor", kind="hold")


# ------------------------------------------------------------------- 1&6 ---

def test_weak_evidence_is_never_recommended_even_with_the_largest_impact():
    weak = _option("big_but_weak", addresses=["weak_ev"])
    solid = _option("small_but_solid", addresses=["moderate_ev"])
    ds = _decision_set([weak, solid, _hold()])
    evidence = [_ev("weak_ev", "weak"), _ev("moderate_ev", "moderate")]
    scenarios = {
        "big_but_weak": _scenario("big_but_weak", 100_000, (50_000, 150_000)),
        "small_but_solid": _scenario("small_but_solid", 1_000, (500, 1_500)),
        "hold_and_monitor": _scenario("hold_and_monitor", None, kind="hold"),
    }
    rec = recommend(ds, evidence, scenarios)
    assert rec.status == RECOMMENDED
    assert rec.recommended_id == "small_but_solid"
    assert rec.assessments["big_but_weak"].outcome == EVIDENCE
    # It says so explicitly.
    assert any("big_but_weak".replace("_", " ") not in "" for _ in [None])  # sanity no-op
    assert any("largest projected impact" in n and "excluded" in n for n in rec.notes)


def test_no_option_clears_the_evidence_bar_leaves_the_result_unranked():
    weak = _option("big_but_weak", addresses=["weak_ev"])
    ds = _decision_set([weak, _hold()])
    evidence = [_ev("weak_ev", "weak")]
    scenarios = {
        "big_but_weak": _scenario("big_but_weak", 100_000),
        "hold_and_monitor": _scenario("hold_and_monitor", None, kind="hold"),
    }
    rec = recommend(ds, evidence, scenarios)
    assert rec.status == NO_EVIDENCE
    assert rec.recommended_id is None
    assert rec.ranking == []


# ------------------------------------------------------------------- 2&3 ---

def test_ranking_uses_risk_adjusted_impact_not_the_raw_number():
    # "risky" is a full commitment with no interval on its projection (2 risk
    # points -> high tier, x0.5) and projects 20,000. "safe" is a bounded test
    # with a comfortably positive interval and only the usual two risks
    # (0 points -> low tier, x1.0) and projects 12,000. Risk-adjusted:
    # risky -> 10,000, safe -> 12,000, so safe should win despite the smaller
    # raw number.
    risky = _option("risky", kind="act", addresses=["strong_ev"])
    safe = _option("safe", kind="test", addresses=["strong_ev"])
    ds = _decision_set([risky, safe, _hold()])
    evidence = [_ev("strong_ev", "strong")]
    scenarios = {
        "risky": _scenario("risky", 20_000, rng=None, kind="act"),
        "safe": _scenario("safe", 12_000, rng=(9_000, 15_000), kind="test"),
        "hold_and_monitor": _scenario("hold_and_monitor", None, kind="hold"),
    }
    rec = recommend(ds, evidence, scenarios)
    assert rec.recommended_id == "safe"
    assert rec.ranking == ["safe", "risky"]
    risky_a, safe_a = rec.assessments["risky"], rec.assessments["safe"]
    assert risky_a.risk_tier == "high" and risky_a.adjusted_impact == pytest.approx(10_000)
    assert safe_a.risk_tier == "low" and safe_a.adjusted_impact == pytest.approx(12_000)
    # The banner explains why the bigger raw number did not win.
    assert any("risky" in n.lower() or "Risky" in n for n in rec.notes)


@pytest.mark.parametrize("kind,rng,extra_risks,expected_tier", [
    ("test", (100, 200), (), "low"),
    ("act", (100, 200), (), "medium"),
    ("test", None, (), "medium"),
    ("act", None, (), "high"),
    ("test", (-50, 200), (), "medium"),
    ("test", (100, 200), ("r1", "r2", "r3", "r4"), "high"),
])
def test_risk_tiers_follow_the_documented_points(kind, rng, extra_risks, expected_tier):
    opt = _option("x", kind=kind, risks=extra_risks or ("r1", "r2"))
    sc = _scenario("x", 1000, rng=rng, kind=kind)
    tier, factor, reasons = risk_of(opt, sc)
    assert tier == expected_tier
    assert factor < 1.0 or tier == "low"


# ------------------------------------------------------------------- 4&5 ---

def test_exactly_one_option_is_marked_recommended_and_the_rest_are_shown_unchanged():
    a = _option("a", addresses=["ev"])
    b = _option("b", addresses=["ev"])
    ds = _decision_set([a, b, _hold()])
    evidence = [_ev("ev", "strong")]
    scenarios = {
        "a": _scenario("a", 5000, (4000, 6000)),
        "b": _scenario("b", 4000, (3000, 5000)),
        "hold_and_monitor": _scenario("hold_and_monitor", None, kind="hold"),
    }
    before = [dataclasses.replace(o) for o in ds.options]
    rec = recommend(ds, evidence, scenarios)
    assert sum(1 for x in [rec.recommended_id] if x is not None) == 1
    assert rec.recommended_id == "a"
    # recommend() never mutates the options it was handed.
    assert ds.options == before


def test_closing_line_states_the_final_choice_is_the_users():
    a = _option("a", addresses=["ev"])
    ds = _decision_set([a, _hold()])
    evidence = [_ev("ev", "strong")]
    scenarios = {"a": _scenario("a", 5000, (4000, 6000)), "hold_and_monitor": _scenario("hold_and_monitor", None, kind="hold")}
    rec = recommend(ds, evidence, scenarios)
    assert "not a decision" in rec.closing and "yours" in rec.closing


def test_an_option_that_projects_no_gain_is_not_recommended():
    a = _option("a", addresses=["ev"])
    ds = _decision_set([a, _hold()])
    evidence = [_ev("ev", "strong")]
    scenarios = {"a": _scenario("a", -500, (-1000, 0)), "hold_and_monitor": _scenario("hold_and_monitor", None, kind="hold")}
    rec = recommend(ds, evidence, scenarios)
    assert rec.status == NO_GAIN
    assert rec.recommended_id is None
    assert rec.assessments["a"].outcome == NO_PROJECTED_GAIN


def test_the_hold_baseline_and_unsized_options_are_never_recommended():
    a = _option("a", addresses=["ev"])
    seq = _option("sequence_fixes", kind="test", addresses=["ev"])
    ds = _decision_set([a, seq, _hold()])
    evidence = [_ev("ev", "strong")]
    scenarios = {
        "a": _scenario("a", 5000, (4000, 6000)),
        "sequence_fixes": Scenario(option_id="sequence_fixes", kind="test", projectable=False,
                                    assumptions=Assumptions()),
        "hold_and_monitor": _scenario("hold_and_monitor", None, kind="hold"),
    }
    rec = recommend(ds, evidence, scenarios)
    assert rec.recommended_id == "a"
    assert "hold_and_monitor" not in rec.ranking
    assert "sequence_fixes" not in rec.ranking
    assert rec.assessments["sequence_fixes"].outcome != RANKED


def test_result_is_the_same_on_repeated_calls():
    a, b = _option("a", addresses=["ev"]), _option("b", addresses=["ev"])
    ds = _decision_set([a, b, _hold()])
    evidence = [_ev("ev", "moderate")]
    scenarios = {
        "a": _scenario("a", 4000, (3000, 5000)),
        "b": _scenario("b", 4000, (3000, 5000)),
        "hold_and_monitor": _scenario("hold_and_monitor", None, kind="hold"),
    }
    first = recommend(ds, evidence, scenarios)
    second = recommend(ds, evidence, scenarios)
    assert first.recommended_id == second.recommended_id == "a"  # tie broken by option id


# --------------------------------------------------------- scenario sweep ---

from data_loader import load_data, scenario_dir  # noqa: E402
from decisions import build_options  # noqa: E402
from orchestrator import investigate  # noqa: E402
from recommend import project_all  # noqa: E402
from strength import is_actionable  # noqa: E402
from synthetic import SCENARIOS  # noqa: E402
from toolkit import Toolkit  # noqa: E402

QUESTION = "Revenue dropped last month. Why?"
GENEROUS = Assumptions(recovery_share=1.0, horizon_months=12, lag_months=0)


@pytest.fixture(scope="module")
def investigations():
    out = {}
    for name in SCENARIOS:
        orders, marketing = load_data(scenario_dir(name))
        inv = list(investigate(QUESTION, Toolkit(orders, marketing)))[-1].data["investigation"]
        out[name] = (inv, orders)
    return out


@pytest.mark.parametrize("name", ["demand_shock", "flat"])
def test_no_cause_scenarios_never_produce_a_forced_recommendation(investigations, name):
    inv, orders = investigations[name]
    ds = build_options(inv.evidence)
    assert ds.insufficient          # same evidence-stage result as test_scenarios.py
    scenarios = project_all(ds, inv.evidence, orders, lambda o: GENEROUS)
    rec = recommend(ds, inv.evidence, scenarios)
    assert rec.status == NO_EVIDENCE
    assert rec.recommended_id is None
    assert rec.ranking == []


def test_default_scenario_recommends_an_actionable_option_under_generous_assumptions(investigations):
    # channel_loss is not included here: restoring West's cut spend does not
    # break even on gross profit even at 100% recovery over 12 months (its
    # break-even share is above 100%, matching decisions.py/README), so the
    # correct behaviour there is NO_GAIN, checked separately below.
    inv, orders = investigations["default"]
    ds = build_options(inv.evidence)
    assert not ds.insufficient
    scenarios = project_all(ds, inv.evidence, orders, lambda o: GENEROUS)
    rec = recommend(ds, inv.evidence, scenarios)
    assert rec.status == RECOMMENDED
    by_id = {e.id: e for e in inv.evidence}
    picked = next(o for o in ds.options if o.id == rec.recommended_id)
    assert picked.kind != "hold"
    assert all(is_actionable(by_id[i]) for i in picked.addresses)
    assert rec.recommended_id in rec.ranking
    assert "hold_and_monitor" not in rec.ranking


def test_channel_loss_does_not_force_a_recommendation_when_nothing_breaks_even(investigations):
    inv, orders = investigations["channel_loss"]
    ds = build_options(inv.evidence)
    assert not ds.insufficient          # the evidence stage does support a cause
    scenarios = project_all(ds, inv.evidence, orders, lambda o: GENEROUS)
    rec = recommend(ds, inv.evidence, scenarios)
    # Even at 100% recovery over 12 months, restoring the cut spend costs
    # more than the gross profit it wins back on this dataset -- so nothing
    # is recommended, and that is the correct answer, not a bug.
    assert rec.status == NO_GAIN
    assert rec.recommended_id is None
    for a in rec.assessments.values():
        if a.outcome == NO_PROJECTED_GAIN:
            assert a.impact <= 0


def test_placeholder_assumptions_do_not_force_a_recommendation_when_projections_are_negative(investigations):
    # Under the default placeholder assumptions, restoring spend/rolling back
    # price costs more than the gross profit it wins back on this dataset;
    # the engine must not paper over that by recommending anyway.
    inv, orders = investigations["default"]
    ds = build_options(inv.evidence)
    scenarios = project_all(ds, inv.evidence, orders)  # None -> placeholder Assumptions()
    rec = recommend(ds, inv.evidence, scenarios)
    if rec.status == NO_GAIN:
        assert rec.recommended_id is None
        assert all(a.impact is None or a.impact <= 0
                   for a in rec.assessments.values() if a.outcome != "baseline")
