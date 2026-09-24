"""Layer 11: sparse or messy data yields "insufficient data" Evidence, not a crash.

Two groups of shapes. The first two sections reproduce what broke on the real
Olist data in Layer 9 (a marketing table that stops before the orders do, and a
long tail of categories with nothing to compare). The rest are synthetic edge
cases: a driver that starts from zero, a region with one or two orders, a channel
with no spend in either month, and comparison groups with no orders.

Grading itself is not under test here. The existing suites already pin what is
strong, moderate and weak when there is enough data; the last section only
checks that nothing they cover was reclassified as insufficient.
"""

import itertools
import math

import pandas as pd
import pytest

from channels import _log_effect, marketing_channel_analysis
from data_loader import load_data, scenario_dir
from decisions import build_options
from effects import (
    InsufficientData, _adjusted_order_effect, _pct, _signed, _undefined_change_reason,
    marketing_effect, price_effect,
)
from narration import narrate_step
from strength import is_insufficient
from synthetic import SCENARIOS
from toolkit import Toolkit

REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Organic", "Paid", "Email"]
CATEGORIES = ["A", "B", "C"]
BASE_SPEND = {"Paid": 1000.0, "Email": 200.0, "Organic": 0.0}


def world(base=40, order_factor=None, spend_factor=None, price_factor=None):
    """A noise-free two-month world. Each (region, channel, category) cell has
    `base` orders in January. `order_factor[region]` scales February orders,
    `spend_factor[(region, channel)]` scales February spend, and
    `price_factor[category]` scales February unit prices."""
    order_factor = order_factor or {}
    spend_factor = spend_factor or {}
    price_factor = price_factor or {}
    rows, spend_rows = [], []
    for region, channel, category in itertools.product(REGIONS, CHANNELS, CATEGORIES):
        for month, n, price in (
            ("2026-01-15", base, 10.0),
            ("2026-02-15", round(base * order_factor.get(region, 1.0)), 10.0 * price_factor.get(category, 1.0)),
        ):
            rows += [(month, region, category, channel, price, price)] * n
    for region, channel in itertools.product(REGIONS, CHANNELS):
        s = BASE_SPEND[channel]
        spend_rows.append(("2026-01-15", channel, region, s, 1000))
        spend_rows.append(("2026-02-15", channel, region, s * spend_factor.get((region, channel), 1.0), 1000))
    orders = pd.DataFrame(
        rows, columns=["order_date", "region", "category", "channel", "revenue", "unit_price"]
    )
    orders["product_id"] = "P_" + orders["category"]
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    marketing = pd.DataFrame(spend_rows, columns=["date", "channel", "region", "spend", "impressions"])
    marketing["date"] = pd.to_datetime(marketing["date"])
    return orders, marketing


def by_id(evidence):
    return {e.id: e for e in evidence}


def assert_is_insufficient(ev, reason=None):
    assert is_insufficient(ev)
    assert ev.strength == "weak"
    assert ev.value is None
    assert ev.details["ci_low_pct"] is None and ev.details["p_value"] is None
    assert ev.details.get("revenue_at_stake") is None
    assert ev.caveats
    assert "insufficient data" in ev.hypothesis or ev.details["insufficient_reason"] == "no_comparison_group"
    if reason:
        assert ev.details["insufficient_reason"] == reason


def assert_not_insufficient(ev):
    assert not is_insufficient(ev)
    assert ev.value is not None


# ---------------------------------------------- Layer 9: marketing table ends early

@pytest.mark.parametrize("keep", ["january_only", "nothing_after_december", "empty"])
def test_marketing_table_that_does_not_reach_the_compared_months(keep):
    # Olist shape: the funnel data stopped before the orders did. Absent rows say
    # nothing about spend; they must not be read as "spend fell to zero".
    orders, marketing = world()
    if keep == "january_only":
        marketing = marketing[marketing["date"].dt.month == 1]
    elif keep == "nothing_after_december":
        marketing = marketing.assign(date=marketing["date"] - pd.DateOffset(months=3))
    else:
        marketing = marketing.iloc[0:0]

    evs = marketing_effect(orders, marketing)   # used to raise KeyError
    assert {e.id for e in evs} == {f"stat_marketing_{r}" for r in REGIONS}
    for e in evs:
        assert_is_insufficient(e, "no_marketing_data")
        assert "no rows for" in e.details["insufficient_reason_text"]
    # Orders are still counted, so the report can show what was untestable.
    assert all(e.sample_size > 0 for e in evs)


def test_marketing_gap_does_not_stop_the_other_tools_through_the_toolkit():
    orders, marketing = world()
    tk = Toolkit(orders, marketing[marketing["date"].dt.month == 1])
    for name in ("marketing_effect", "marketing_channel_analysis", "price_effect"):
        evs = tk.run(name, {})
        assert all(e.details["provenance"]["tool"] == name for e in evs)


# --------------------------------------------- Layer 9: sparse tail of categories

def test_category_with_no_product_sold_in_both_months():
    # Olist shape: a long-tail category whose products in one month are all new.
    orders, marketing = world()
    orders = orders.copy()
    is_c_last = (orders["category"] == "C") & (orders["order_date"].dt.month == 2)
    orders.loc[is_c_last, "product_id"] = "P_C_new"

    evs = by_id(price_effect(orders))          # used to raise ValueError (NaN -> int)
    assert_is_insufficient(evs["stat_price_C"], "no_price_baseline")
    assert "like-for-like" in evs["stat_price_C"].details["insufficient_reason_text"]
    assert evs["stat_price_C"].details["driver_change_pct"] is None
    # The categories that can be tested still are.
    for cat in ("A", "B"):
        assert_not_insufficient(evs[f"stat_price_{cat}"])


def test_category_sold_only_in_one_of_the_two_months():
    orders, _ = world()
    orders = orders[~((orders["category"] == "C") & (orders["order_date"].dt.month == 2))]
    evs = by_id(price_effect(orders))
    assert_is_insufficient(evs["stat_price_C"], "no_price_baseline")
    assert evs["stat_price_C"].sample_size == 0


def test_zero_prior_price_has_no_percent_change():
    # A prior price of zero would divide by zero (and give inf, then a rounding error).
    orders, _ = world()
    orders = orders.copy()
    orders.loc[(orders["category"] == "C") & (orders["order_date"].dt.month == 1), "unit_price"] = 0.0
    evs = by_id(price_effect(orders))
    assert_is_insufficient(evs["stat_price_C"], "no_price_baseline")
    for cat in ("A", "B"):
        assert_not_insufficient(evs[f"stat_price_{cat}"])


def test_every_category_untestable_still_returns_one_evidence_per_category():
    orders, _ = world()
    orders = orders.copy()
    orders.loc[orders["order_date"].dt.month == 2, "product_id"] += "_new"
    evs = price_effect(orders)
    assert len(evs) == len(CATEGORIES)
    for e in evs:
        assert_is_insufficient(e, "no_price_baseline")


# --------------------------------------------------------- spend starts from zero

def test_region_with_zero_spend_in_the_prior_month():
    orders, marketing = world()
    marketing = marketing.copy()
    marketing.loc[(marketing["region"] == "North") & (marketing["date"].dt.month == 1), "spend"] = 0.0

    for evs in (by_id(marketing_effect(orders, marketing)),):
        assert_is_insufficient(evs["stat_marketing_North"], "no_driver_baseline")
        assert "undefined" in evs["stat_marketing_North"].details["insufficient_reason_text"]
        # The other regions are compared as before, just without North to serve as a control.
        assert evs["stat_marketing_South"].details["control_segments"]
        assert "North" not in evs["stat_marketing_South"].details["control_segments"]
    chans = marketing_channel_analysis(orders, marketing)   # used to raise ValueError
    assert {e.details["insufficient_reason"] for e in chans if is_insufficient(e)} == {"no_driver_baseline"}


def test_region_with_no_spend_at_all_in_either_month():
    orders, marketing = world()
    marketing = marketing.copy()
    marketing.loc[marketing["region"] == "North", "spend"] = 0.0
    evs = by_id(marketing_effect(orders, marketing))
    assert_is_insufficient(evs["stat_marketing_North"], "no_driver_in_either_period")
    assert "zero in both" in evs["stat_marketing_North"].details["insufficient_reason_text"]


# ----------------------------------------------- channel with zero spend both months

def test_channel_with_zero_spend_in_both_compared_months():
    # Email is paid everywhere else, but North recorded nothing for it in either month.
    orders, marketing = world(order_factor={"North": 0.5}, spend_factor={("North", "Paid"): 0.55})
    marketing = marketing.copy()
    marketing.loc[(marketing["region"] == "North") & (marketing["channel"] == "Email"), "spend"] = 0.0

    evs = by_id(marketing_channel_analysis(orders, marketing))   # used to raise ValueError
    email_north = evs["chan_North_Email"]
    assert_is_insufficient(email_north, "no_driver_in_either_period")
    assert email_north.details["driver_change_pct"] is None
    assert email_north.details["orders_prior"] > 0            # the orders are real; the spend is not
    # The genuinely cut channel in the same region is still found.
    assert evs["chan_North_Paid"].strength == "strong"
    assert evs["chan_North_Paid"].details["channel_pattern"] is not None


def test_channel_with_zero_spend_everywhere_is_left_out_as_before():
    # Organic has no spend in any region, so it is not a paid channel and gets no cell.
    orders, marketing = world()
    evs = marketing_channel_analysis(orders, marketing)
    assert {e.details["channel"] for e in evs} == {"Paid", "Email"}


def test_channel_analysis_with_no_spend_in_either_month_anywhere_returns_nothing():
    orders, marketing = world()
    marketing = marketing.assign(spend=0.0)
    assert marketing_channel_analysis(orders, marketing) == []


# ------------------------------------------------------------ tiny and empty groups

def test_region_with_one_or_two_orders_in_the_latest_month():
    # No crash, and a sample this small can never come out strong: the existing
    # small-sample rule lowers it one level and says so.
    orders, marketing = world(spend_factor={("North", "Paid"): 0.4, ("North", "Email"): 0.4})
    feb_north = (orders["region"] == "North") & (orders["order_date"].dt.month == 2)
    orders = pd.concat([orders[~feb_north], orders[feb_north].iloc[:2]])

    evs = by_id(marketing_effect(orders, marketing))
    north = evs["stat_marketing_North"]
    assert north.sample_size == 2
    assert north.strength != "strong"
    assert any("sample size (2)" in c for c in north.caveats)
    chans = marketing_channel_analysis(orders, marketing)
    assert all(e.strength != "strong" for e in chans if e.details["region"] == "North")


def test_region_with_one_or_two_orders_in_both_months_is_weak():
    orders, marketing = world()
    north = orders["region"] == "North"
    keep = orders[north].groupby(orders["order_date"].dt.month).head(2)
    orders = pd.concat([orders[~north], keep])
    evs = by_id(marketing_effect(orders, marketing))
    assert evs["stat_marketing_North"].strength == "weak"
    for e in marketing_channel_analysis(orders, marketing):
        assert e.strength == "weak" or e.details["region"] != "North"


def test_region_with_orders_in_neither_compared_month():
    orders, marketing = world()
    old = orders[orders["region"] == "North"].copy()
    old["order_date"] = pd.Timestamp("2025-12-15")
    orders = pd.concat([orders[orders["region"] != "North"], old])
    evs = by_id(marketing_effect(orders, marketing))
    assert_is_insufficient(evs["stat_marketing_North"], "no_orders_in_segment")
    chans = [e for e in marketing_channel_analysis(orders, marketing) if e.details["region"] == "North"]
    assert chans and all(is_insufficient(e) for e in chans)


def test_comparison_group_with_no_orders():
    # Only North sold anything in the compared months. Every other region is
    # "typical" on spend but has no orders to compare North against.
    orders, marketing = world(spend_factor={("North", "Paid"): 0.4, ("North", "Email"): 0.4})
    others = orders["region"] != "North"
    old = orders[others].copy()
    old["order_date"] = pd.Timestamp("2025-12-15")
    orders = pd.concat([orders[~others], old])
    evs = by_id(marketing_effect(orders, marketing))
    assert_is_insufficient(evs["stat_marketing_North"], "no_orders_in_comparison_group")
    assert "comparison group" in evs["stat_marketing_North"].details["insufficient_reason_text"]


def test_no_categories_to_compare_within():
    orders, _ = world()
    orders = orders.assign(category=None)
    seg = orders["region"] == "North"
    ctl = orders["region"] != "North"
    period = orders["order_date"].dt.to_period("M")
    with pytest.raises(InsufficientData) as exc:
        _adjusted_order_effect(orders, seg, ctl, "category", period, period.max(), period.min())
    assert exc.value.code == "no_strata"


# ------------------------------------------------------------ arithmetic helpers

def test_percent_change_against_a_zero_or_near_zero_baseline_is_undefined():
    assert math.isnan(_pct(10.0, 0.0))
    assert math.isnan(_pct(10.0, 1e-12))
    assert math.isnan(_pct(float("nan"), 5.0))
    assert math.isnan(_pct(5.0, float("inf")))
    assert _pct(110.0, 100.0) == pytest.approx(10.0)
    assert _pct(0.0, 100.0) == pytest.approx(-100.0)     # spend that really fell to zero is a real change


def test_signed_never_raises_on_a_non_finite_value():
    assert _signed(float("nan")) == "n/a"
    assert _signed(float("inf")) == "n/a"
    assert _signed(-0.2) == "+0%"
    assert _signed(-12.4) == "-12%"


def test_undefined_change_reason():
    assert _undefined_change_reason("spend", 0.0, 0.0, "a", "b")[0] == "no_driver_in_either_period"
    assert _undefined_change_reason("spend", 0.0, 5.0, "a", "b")[0] == "no_driver_baseline"
    assert _undefined_change_reason("spend", 5.0, 0.0, "a", "b") is None      # a real -100% move


def test_log_effect_refuses_a_percentage_it_cannot_take_the_log_of():
    with pytest.raises(InsufficientData):
        _log_effect(-100.0, -100.0, -60.0)
    with pytest.raises(InsufficientData):
        _log_effect(-40.0, -100.0, -10.0)


# --------------------------------------------------- what downstream says about it

def test_untestable_causes_are_not_reported_as_tested_and_not_supported():
    orders, marketing = world()
    marketing = marketing[marketing["date"].dt.month == 1]        # marketing cannot be tested at all
    orders = orders.copy()
    orders.loc[(orders["category"] == "C") & (orders["order_date"].dt.month == 2), "product_id"] = "P_C_new"
    evidence = marketing_effect(orders, marketing) + price_effect(orders)
    decision = build_options(evidence)

    joined = " ".join(decision.not_supported)
    assert "North" not in joined                                   # untestable, so not "not supported"
    assert not any(x.startswith("Marketing spend") for x in decision.not_supported)
    assert "Pricing: A, B" in joined and "C" not in joined.split("Pricing:")[1]
    note = next(x for x in decision.unresolved if x.startswith("Could not be tested"))
    assert "Marketing spend: East, North, South, West" in note
    assert "Pricing: C" in note
    assert decision.insufficient                                    # no supported cause


def test_narration_says_untestable_not_unsupported():
    orders, marketing = world()
    evs = marketing_effect(orders, marketing[marketing["date"].dt.month == 1])
    text = narrate_step("marketing_effect", {}, evs)
    assert "not sufficient" in text and "neither supported nor ruled out" in text
    assert "is not supported as an explanation" not in text

    # A mix: what could be tested is reported as before, and the rest is named as untested.
    orders, _ = world()
    orders = orders.copy()
    orders.loc[(orders["category"] == "C") & (orders["order_date"].dt.month == 2), "product_id"] = "P_C_new"
    text = narrate_step("price_effect", {}, price_effect(orders))
    assert "could be tested" in text and "insufficient data" in text


# ------------------------------------------- nothing that was gradable has changed

@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_committed_scenarios_never_produce_insufficient_evidence(scenario):
    orders, marketing = load_data(scenario_dir(scenario))
    evs = marketing_effect(orders, marketing) + price_effect(orders) + marketing_channel_analysis(orders, marketing)
    assert evs
    assert not [e.id for e in evs if is_insufficient(e)]
