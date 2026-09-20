"""Layer 3: fewer orders vs. smaller orders."""

import dataclasses

import pandas as pd
import pytest

from decisions import build_options
from decomposition import MIN_HISTORY, aov_volume_decomposition, signature_check
from effects import marketing_effect, price_effect
from guardrail import check_text
from narration import build_summary, metric_note, narrate_step
from toolkit import TOOL_SPECS, ToolError, Toolkit


def frame(rows):
    """rows: (date, region, revenue) -> minimal orders frame."""
    return pd.DataFrame(rows, columns=["order_date", "region", "revenue"]).assign(
        order_date=lambda d: pd.to_datetime(d["order_date"])
    )


def by_id(evidence):
    return {e.id: e for e in evidence}


# ------------------------------------------------------------- the arithmetic

def test_hand_computed_split():
    # Prior month: 10 orders at 100 = 1,000. Latest: 5 orders at 120 = 600.
    rows = [("2026-01-05", "A", 100.0)] * 10 + [("2026-02-05", "A", 120.0)] * 5
    ev = by_id(aov_volume_decomposition(frame(rows)))
    o, a = ev["decomp_orders_overall"], ev["decomp_aov_overall"]
    # volume: (5 - 10) * (100 + 120) / 2 = -550; AOV: (120 - 100) * (10 + 5) / 2 = +150
    assert o.details["orders_effect"] == -550.0 and a.details["aov_effect"] == 150.0
    assert o.details["orders_effect_pts"] == -55.0 and a.details["aov_effect_pts"] == 15.0
    assert o.details["revenue_change_pct"] == -40.0
    assert o.value == -50.0 and a.value == 20.0


def test_two_effects_add_up_to_the_revenue_change_exactly(orders):
    for dim in (None, "region", "category"):
        evs = aov_volume_decomposition(orders, dim)
        for e in evs:
            d = e.details
            assert d["orders_effect"] + d["aov_effect"] == pytest.approx(d["revenue_last"] - d["revenue_prior"], abs=0.02)
            assert d["orders_effect_pts"] + d["aov_effect_pts"] == pytest.approx(d["revenue_change_pct"], abs=0.11)


def test_two_hypotheses_per_group(orders):
    evs = aov_volume_decomposition(orders, "region")
    assert len(evs) == 2 * orders["region"].nunique()
    assert {e.metric for e in evs} == {"orders_change_pct", "aov_change_pct"}
    assert all(e.evidence_type == "decomposition" for e in evs)


# ------------------------------------------------------- what the data shows

def test_overall_change_is_fewer_orders_not_smaller_orders(orders):
    ev = by_id(aov_volume_decomposition(orders))
    assert ev["decomp_orders_overall"].strength in ("strong", "moderate")
    assert ev["decomp_orders_overall"].value < 0
    assert ev["decomp_aov_overall"].strength == "weak"
    assert ev["decomp_orders_overall"].details["leading_factor"] == "orders"


def test_north_is_a_pure_volume_loss(orders):
    ev = by_id(aov_volume_decomposition(orders, "region"))
    assert ev["decomp_orders_region_North"].strength == "strong"
    assert ev["decomp_aov_region_North"].strength == "weak"


def test_electronics_shows_fewer_orders_and_higher_order_value(orders):
    ev = by_id(aov_volume_decomposition(orders, "category"))
    o, a = ev["decomp_orders_category_Electronics"], ev["decomp_aov_category_Electronics"]
    assert o.strength != "weak" and o.value < 0
    assert a.strength != "weak" and a.value > 0


def test_unaffected_segments_stay_weak(orders):
    evs = aov_volume_decomposition(orders, "region") + aov_volume_decomposition(orders, "category")
    flagged = {e.id for e in evs if e.strength != "weak"}
    assert flagged == {
        "decomp_orders_region_North", "decomp_orders_category_Electronics",
        "decomp_aov_category_Electronics",
    }


def test_no_strong_findings_in_earlier_months(orders):
    """Nothing was injected before the final month. At a 5% bar a few moderate
    flags are expected by chance; strong ones and a flood are not."""
    period = orders["order_date"].dt.to_period("M")
    months = sorted(period.unique())
    total = flagged = 0
    for cutoff in months[MIN_HISTORY + 1:-1]:      # enough earlier months to judge
        sub = orders[period <= cutoff]
        evs = aov_volume_decomposition(sub) + aov_volume_decomposition(sub, "region") \
            + aov_volume_decomposition(sub, "category")
        total += len(evs)
        bad = [e for e in evs if e.strength != "weak"]
        assert not [e for e in bad if e.strength == "strong"]
        flagged += len(bad)
    assert total >= 40 and flagged / total <= 0.10


# ----------------------------------------------------------------- edge cases

def test_too_little_history_is_weak_and_says_so():
    rows = [("2026-01-05", "A", 100.0)] * 40 + [("2026-02-05", "A", 60.0)] * 20
    ev = by_id(aov_volume_decomposition(frame(rows)))
    o = ev["decomp_orders_overall"]
    assert o.strength == "weak"
    assert any("earlier month-on-month changes" in c for c in o.caveats)
    assert o.details["p_value"] is None


def test_a_group_with_no_orders_in_one_month_does_not_crash():
    rows = [("2026-01-05", "A", 50.0)] * 30 + [("2026-01-06", "B", 50.0)] * 30 + [("2026-02-05", "A", 50.0)] * 30
    evs = by_id(aov_volume_decomposition(frame(rows), "region"))
    b = evs["decomp_orders_region_B"]
    assert b.strength == "weak" and b.value is None
    assert "cannot be computed" in b.hypothesis


def test_needs_two_periods():
    with pytest.raises(ValueError):
        aov_volume_decomposition(frame([("2026-01-05", "A", 10.0)]))


def test_short_month_is_flagged(orders):
    feb = orders[orders["order_date"] < "2026-03-01"]
    o = by_id(aov_volume_decomposition(feb))["decomp_orders_overall"]
    assert any("different lengths" in c for c in o.caveats)
    august = by_id(aov_volume_decomposition(orders))["decomp_orders_overall"]
    assert not any("different lengths" in c for c in august.caveats)


# ------------------------------------------------ hypotheses and their pattern

@pytest.fixture(scope="module")
def all_evidence(orders, marketing):
    return (
        aov_volume_decomposition(orders) + aov_volume_decomposition(orders, "region")
        + aov_volume_decomposition(orders, "category")
        + marketing_effect(orders, marketing) + price_effect(orders)
    )


def test_supported_causes_match_the_pattern_they_predict(all_evidence):
    ev = by_id(all_evidence)
    assert signature_check(ev["stat_marketing_North"], all_evidence)["status"] == "consistent"
    assert signature_check(ev["stat_price_Electronics"], all_evidence)["status"] == "consistent"


def test_a_mismatched_pattern_is_reported(all_evidence):
    price = by_id(all_evidence)["stat_price_Electronics"]
    # Pretend the price had been cut: the data would then be the wrong way round.
    cut = dataclasses.replace(price, details={**price.details, "driver_change_pct": -10.0, "driver_typical_change_pct": 0.0})
    assert signature_check(cut, all_evidence)["status"] == "inconsistent"


def test_no_check_without_the_decomposition(orders, marketing):
    cause = marketing_effect(orders, marketing)[0]
    assert signature_check(cause, []) is None


def test_marketing_assumption_is_checked_against_the_data(all_evidence):
    ds = build_options(all_evidence)
    north = next(o for o in ds.options if o.id == "restore_marketing_North")
    assert any("Checked against the data" in a for a in north.assumptions)
    price = next(o for o in ds.options if o.id == "rollback_price_Electronics")
    assert "pattern a price change predicts" in price.rationale


def test_options_still_build_without_the_decomposition(orders, marketing):
    ds = build_options(marketing_effect(orders, marketing) + price_effect(orders))
    north = next(o for o in ds.options if o.id == "restore_marketing_North")
    assert not any("Checked against" in a for a in north.assumptions)


# ------------------------------------------------- tool, narration, guardrail

def test_tool_is_registered_and_stamps_provenance(toolkit):
    assert "aov_volume_decomposition" in {t["name"] for t in TOOL_SPECS}
    overall = toolkit.run("aov_volume_decomposition")
    assert overall[0].details["provenance"] == {
        "tool": "aov_volume_decomposition", "args": {}, "source": "src/decomposition.py",
    }
    region = toolkit.run("aov_volume_decomposition", {"dimension": "region"})
    assert region[0].details["provenance"]["args"] == {"dimension": "region"}


def test_tool_rejects_an_unknown_dimension(toolkit):
    with pytest.raises(ToolError):
        toolkit.run("aov_volume_decomposition", {"dimension": "colour"})


@pytest.mark.parametrize("args", [{}, {"dimension": "region"}, {"dimension": "category"}])
def test_offline_narration_passes_the_numeric_guardrail(toolkit, args):
    evs = toolkit.run("aov_volume_decomposition", args)
    text = narrate_step("aov_volume_decomposition", args, evs)
    assert check_text(text, evs).ok, text


def test_narration_reads_correctly(toolkit):
    overall = narrate_step("aov_volume_decomposition", {}, toolkit.run("aov_volume_decomposition"))
    assert "number of orders" in overall and "not in order size" in overall
    region = narrate_step("aov_volume_decomposition", {"dimension": "region"},
                          toolkit.run("aov_volume_decomposition", {"dimension": "region"}))
    assert "North" in region and "no clear evidence it moved" in region
    assert "Electronics" in narrate_step(
        "aov_volume_decomposition", {"dimension": "category"},
        toolkit.run("aov_volume_decomposition", {"dimension": "category"}),
    )


def test_summary_states_fewer_or_smaller(all_evidence):
    summary = build_summary(all_evidence)
    assert "Overall (vs. the prior month): the change is in the number of orders" in summary
    assert check_text(summary, all_evidence).ok


def test_metric_note_distinguishes_the_two_hypotheses(orders):
    o, a = aov_volume_decomposition(orders)[:2]
    assert (metric_note(o), metric_note(a)) == ("orders", "order value")
