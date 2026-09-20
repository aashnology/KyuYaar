"""
Scenario engine tests.

Most cases use small constructed evidence and orders where the answer can be
worked out on paper, so a wrong formula cannot hide behind realistic-looking
numbers. The last group runs the engine on the real investigation.
"""

import pandas as pd
import pytest

from decisions import DecisionOption, build_options
from evidence import Evidence
from orchestrator import investigate
from report import build_report
from scenario import Assumptions, gross_margin, run_scenario


# ---------------------------------------------------------- constructions ---

def _orders(with_cost=True):
    """North: 100 revenue at 60 cost in June, 200 at 100 in July.
    Margins are 40% (June) and 50% (July)."""
    rows = [
        ("2026-06-10", "North", 4, 100.0, 15.0),
        ("2026-07-10", "North", 5, 200.0, 20.0),
        ("2026-07-11", "South", 1, 999.0, 1.0),
    ]
    df = pd.DataFrame(rows, columns=["order_date", "region", "quantity", "revenue", "unit_cost"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df if with_cost else df.drop(columns="unit_cost")


def _marketing_ev(stake=1000.0, cut=300.0, ci=(-50.0, -30.0), prior_revenue=2500.0):
    return Evidence(
        id="stat_marketing_North", hypothesis="h", evidence_type="statistical",
        metric="orders_change_vs_comparable_segments_pct", value=-40.0, baseline=0.0,
        segment="region=North", strength="strong", sample_size=200,
        details={
            "period": "2026-07", "prior_period": "2026-06",
            "spend_prior": 1000.0, "spend_last": 1000.0 - cut,
            "revenue_prior": prior_revenue, "revenue_last": 1500.0,
            "revenue_at_stake": stake, "ci_low_pct": ci[0], "ci_high_pct": ci[1],
        },
    )


def _price_ev(stake=200.0, revenue_last=1000.0):
    return Evidence(
        id="stat_price_Electronics", hypothesis="h", evidence_type="statistical",
        metric="orders_change_vs_comparable_segments_pct", value=-20.0, baseline=0.0,
        segment="category=Electronics", strength="strong", sample_size=200,
        details={
            "period": "2026-07", "prior_period": "2026-06",
            "driver_change_pct": 10.0, "driver_typical_change_pct": 0.0,
            "revenue_prior": 1200.0, "revenue_last": revenue_last,
            "revenue_at_stake": stake, "ci_low_pct": -30.0, "ci_high_pct": -10.0,
        },
    )


def _price_orders():
    """Electronics: 35% margin in June, 40% in July."""
    rows = [
        ("2026-06-10", "Electronics", 1, 1000.0, 650.0),
        ("2026-07-10", "Electronics", 1, 1000.0, 600.0),
    ]
    df = pd.DataFrame(rows, columns=["order_date", "category", "quantity", "revenue", "unit_cost"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def _option(option_id, kind, addresses):
    return DecisionOption(
        id=option_id, kind=kind, title=option_id, rationale="", addresses=addresses,
        confidence="strong", impact="", assumptions=[], risks=[],
    )


ACT = lambda ev: _option("restore_marketing_North", "act", [ev.id])
TEST = lambda ev: _option("test_marketing_North", "test", [ev.id])


# ------------------------------------------------------------ gross margin ---

def test_gross_margin_is_one_minus_cost_over_revenue():
    o = _orders()
    assert gross_margin(o, "region=North", "2026-06") == pytest.approx(0.40)
    assert gross_margin(o, "region=North", "2026-07") == pytest.approx(0.50)


def test_gross_margin_is_none_without_cost_or_sales():
    assert gross_margin(_orders(with_cost=False), "region=North", "2026-06") is None
    assert gross_margin(_orders(), "region=North", "2025-01") is None
    assert gross_margin(None, "region=North", "2026-06") is None


# --------------------------------------------------------------- marketing ---

def test_marketing_act_matches_the_hand_calculation():
    ev = _marketing_ev()
    a = Assumptions(recovery_share=0.6, lag_months=1, horizon_months=4)
    sc = run_scenario(ACT(ev), [ev], _orders(), a)
    # 3 months pay off: 1000 x 0.6 x 3 = 1800. Gross profit 1800 x 0.40 = 720.
    # Spend restored: 300 a month for 4 months = 1200.
    assert sc.revenue_total == pytest.approx(1800.0)
    assert sc.added_cost_total == pytest.approx(1200.0)
    assert sc.gross_profit_total == pytest.approx(720.0 - 1200.0)
    assert sc.break_even_share == pytest.approx(1200.0 / (1000.0 * 0.40 * 3))


def test_gross_profit_is_zero_at_the_break_even_share():
    ev = _marketing_ev()
    base = Assumptions(recovery_share=0.5, lag_months=1, horizon_months=4)
    be = run_scenario(ACT(ev), [ev], _orders(), base).break_even_share
    at_be = Assumptions(recovery_share=be, lag_months=1, horizon_months=4)
    assert run_scenario(ACT(ev), [ev], _orders(), at_be).gross_profit_total == pytest.approx(0.0, abs=1e-6)


def test_more_recovery_never_lowers_revenue_or_profit():
    ev = _marketing_ev()
    prev = None
    for share in (0.0, 0.25, 0.5, 0.75, 1.0):
        sc = run_scenario(ACT(ev), [ev], _orders(), Assumptions(recovery_share=share))
        if prev:
            assert sc.revenue_total >= prev.revenue_total
            assert sc.gross_profit_total >= prev.gross_profit_total
        prev = sc


def test_zero_recovery_recovers_nothing_but_still_costs_the_spend():
    ev = _marketing_ev()
    sc = run_scenario(ACT(ev), [ev], _orders(), Assumptions(recovery_share=0.0, horizon_months=3))
    assert sc.revenue_total == 0.0
    assert sc.gross_profit_total == pytest.approx(-sc.added_cost_total)


def test_a_lag_as_long_as_the_horizon_leaves_no_revenue_and_says_so():
    ev = _marketing_ev()
    sc = run_scenario(ACT(ev), [ev], _orders(), Assumptions(lag_months=3, horizon_months=3))
    assert sc.revenue_total == 0.0
    assert "lag uses up the whole horizon" in sc.summary
    assert sc.break_even_share is None


def test_test_option_scales_revenue_cost_and_profit_by_the_treated_share():
    ev = _marketing_ev()
    a = Assumptions(recovery_share=0.6, test_share=0.25)
    act = run_scenario(ACT(ev), [ev], _orders(), a)
    test = run_scenario(TEST(ev), [ev], _orders(), a)
    assert test.revenue_total == pytest.approx(act.revenue_total * 0.25)
    assert test.added_cost_total == pytest.approx(act.added_cost_total * 0.25)
    assert test.gross_profit_total == pytest.approx(act.gross_profit_total * 0.25)
    assert test.break_even_share == pytest.approx(act.break_even_share)


def test_range_comes_from_the_interval_and_brackets_the_point():
    ev = _marketing_ev(stake=1000.0, ci=(-50.0, -30.0), prior_revenue=2500.0)
    sc = run_scenario(ACT(ev), [ev], _orders(), Assumptions(recovery_share=1.0, lag_months=0, horizon_months=1))
    # Stake at each end of the interval: 2500 x 30% = 750 and 2500 x 50% = 1250.
    assert sc.revenue_range == pytest.approx((750.0, 1250.0))
    assert sc.revenue_range[0] <= sc.revenue_total <= sc.revenue_range[1]


def test_no_interval_means_no_range():
    ev = _marketing_ev()
    ev.details.pop("ci_low_pct")
    sc = run_scenario(ACT(ev), [ev], _orders())
    assert sc.revenue_range is None and sc.gross_profit_range is None


def test_without_product_cost_revenue_is_projected_and_profit_is_left_out():
    ev = _marketing_ev()
    sc = run_scenario(ACT(ev), [ev], _orders(with_cost=False))
    assert sc.revenue_total is not None
    assert sc.gross_profit_total is None and sc.break_even_share is None
    assert all("Gross" not in s.label for s in sc.steps)


# ------------------------------------------------------------------- price ---

def test_price_rollback_matches_the_hand_calculation():
    ev = _price_ev(stake=200.0, revenue_last=1000.0)
    opt = _option("rollback_price_Electronics", "act", [ev.id])
    a = Assumptions(recovery_share=1.0, lag_months=1, horizon_months=3)
    sc = run_scenario(opt, [ev], _price_orders(), a)
    # After: 35% x (1000 + 200) = 420. Now: 40% x 1000 = 400. +20 a month for 2 months.
    assert sc.revenue_total == pytest.approx(200.0 * 2)
    assert sc.gross_profit_total == pytest.approx(40.0)
    assert sc.added_cost_total is None       # the margin effect is inside gross profit


def test_price_rollback_loses_profit_when_demand_does_not_return():
    ev = _price_ev()
    opt = _option("rollback_price_Electronics", "act", [ev.id])
    sc = run_scenario(opt, [ev], _price_orders(), Assumptions(recovery_share=0.0))
    # Margin falls from 40% to 35% on 1000 of revenue and nothing comes back.
    assert sc.gross_profit_total == pytest.approx(-50.0 * 2)


def test_price_break_even_share_is_where_profit_is_zero():
    ev = _price_ev()
    opt = _option("rollback_price_Electronics", "act", [ev.id])
    be = run_scenario(opt, [ev], _price_orders()).break_even_share
    assert be == pytest.approx((0.40 - 0.35) * 1000.0 / (0.35 * 200.0))
    sc = run_scenario(opt, [ev], _price_orders(), Assumptions(recovery_share=be))
    assert sc.gross_profit_total == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------- other option kinds ---

def test_hold_states_what_stays_open_without_adding_overlapping_estimates():
    m, p = _marketing_ev(stake=1000.0), _price_ev(stake=200.0)
    hold = _option("hold_and_monitor", "hold", [m.id, p.id])
    sc = run_scenario(hold, [m, p], _orders())
    assert sc.projectable and sc.revenue_total == 0.0
    assert [s.value for s in sc.steps] == [1000.0, 200.0]
    assert 1200.0 not in [s.value for s in sc.steps]
    assert "not added" in sc.steps[0].formula


def test_hold_with_no_supported_cause_has_nothing_to_size():
    sc = run_scenario(_option("hold_and_monitor", "hold", []), [], _orders())
    assert sc.steps == [] and "nothing to size" in sc.summary


def test_sequence_is_not_projected_because_the_estimates_overlap():
    m, p = _marketing_ev(), _price_ev()
    seq = _option("sequence_fixes", "test", [m.id, p.id])
    sc = run_scenario(seq, [m, p], _orders())
    assert not sc.projectable and sc.revenue_total is None
    assert "double count" in sc.summary


def test_an_option_that_names_no_action_is_not_projected():
    ev = _price_ev()
    sc = run_scenario(_option("review_price_Electronics", "act", [ev.id]), [ev], _price_orders())
    assert not sc.projectable


# --------------------------------------------------------------- validation ---

@pytest.mark.parametrize("kwargs", [
    {"recovery_share": 1.2}, {"recovery_share": -0.1}, {"test_share": 0.0},
    {"horizon_months": 0}, {"horizon_months": 25}, {"lag_months": -1},
    {"horizon_months": 3, "lag_months": 4},
])
def test_invalid_assumptions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Assumptions(**kwargs)


# ----------------------------------------------------------- real dataset ---

@pytest.fixture(scope="module")
def inv(toolkit):
    return [e for e in investigate("why", toolkit) if e.kind == "done"][0].data["investigation"]


@pytest.fixture(scope="module")
def options(inv):
    return {o.id: o for o in build_options(inv.evidence).options}


def _by_id(inv):
    return {e.id: e for e in inv.evidence}


def test_every_option_gets_a_scenario_and_only_sequence_is_declined(inv, options, orders):
    for opt in options.values():
        sc = run_scenario(opt, inv.evidence, orders)
        assert sc.projectable == (opt.id != "sequence_fixes")


def test_north_projection_uses_the_evidence_stake_and_an_independent_margin(inv, options, orders, data):
    ev = _by_id(inv)["stat_marketing_North"]
    a = Assumptions(recovery_share=0.8, lag_months=1, horizon_months=6)
    sc = run_scenario(options["restore_marketing_North"], inv.evidence, orders, a)

    stake = ev.details["revenue_at_stake"]
    assert sc.revenue_total == pytest.approx(stake * 0.8 * 5)

    # Margin recomputed straight from the raw CSVs, not through gross_margin().
    from pathlib import Path
    raw = Path(__file__).resolve().parents[1] / "data"
    o = pd.read_csv(raw / "orders.csv", parse_dates=["order_date"])
    p = pd.read_csv(raw / "products.csv")
    c = pd.read_csv(raw / "customers.csv")
    j = o.merge(p, on="product_id").merge(c[["customer_id", "region"]], on="customer_id")
    j = j[(j["region"] == "North") & (j["order_date"].dt.strftime("%Y-%m") == ev.details["prior_period"])]
    margin = 1 - (j["cost"] * j["quantity"]).sum() / j["revenue"].sum()
    cut = ev.details["spend_prior"] - ev.details["spend_last"]
    assert sc.gross_profit_total == pytest.approx(sc.revenue_total * margin - cut * 6)


def test_electronics_break_even_is_where_profit_is_zero(inv, options, orders):
    opt = options["rollback_price_Electronics"]
    be = run_scenario(opt, inv.evidence, orders).break_even_share
    assert 0 < be < 1
    at_be = run_scenario(opt, inv.evidence, orders, Assumptions(recovery_share=be))
    assert at_be.gross_profit_total == pytest.approx(0.0, abs=1e-6)


def test_the_last_step_totals_agree_with_the_scenario(inv, options, orders):
    sc = run_scenario(options["restore_marketing_North"], inv.evidence, orders)
    by_label = {s.label: s.value for s in sc.steps}
    assert by_label["Revenue recovered over 3 months"] == pytest.approx(sc.revenue_total)
    assert by_label["Gross profit after the added spend"] == pytest.approx(sc.gross_profit_total)
    assert all(s.formula for s in sc.steps)


def test_projection_is_deterministic(inv, options, orders):
    opt = options["rollback_price_Electronics"]
    a = run_scenario(opt, inv.evidence, orders)
    b = run_scenario(opt, inv.evidence, orders)
    assert a == b


def test_memo_records_projection_assumptions_and_the_chosen_math(inv, options, orders):
    ds = build_options(inv.evidence)
    a = Assumptions(recovery_share=0.7)
    scenarios = {o.id: run_scenario(o, inv.evidence, orders, a) for o in ds.options}
    memo = build_report(inv, ds, "restore_marketing_North", "", scenarios)
    assert "Projected impact" in memo
    assert "wins back 70% of the revenue at stake" in memo
    assert "How the projection for this option was worked out" in memo
    assert "Break-even share on gross profit" in memo
    # Without scenarios the memo is unchanged from Layer 4.
    assert "Projected impact" not in build_report(inv, ds, "restore_marketing_North")
