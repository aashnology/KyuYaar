"""Trend series and charts. The series are checked against the numbers in the findings they illustrate."""

import pytest

from charts import driver_figure, metric_figure
from orchestrator import investigate
from trends import baseline_index, driver_trend, metric_by_group


@pytest.fixture(scope="module")
def evidence(toolkit):
    events = list(investigate("Revenue dropped. Why?", toolkit))
    return {e.id: e for e in events[-1].data["investigation"].evidence}


def last_over_prior(series):
    """The last month against the one before it, as a percentage change."""
    return (series[-1] / series[-2] - 1) * 100


# --------------------------------------------------------- metric series ---

def test_metric_totals_match_the_raw_orders(orders):
    revenue = metric_by_group(orders, "revenue")
    count = metric_by_group(orders, "orders")
    assert list(revenue.columns) == ["All"]
    assert revenue["All"].sum() == pytest.approx(orders["revenue"].sum())
    assert count["All"].sum() == len(orders)


def test_average_order_value_is_revenue_over_orders(orders):
    aov = metric_by_group(orders, "aov")["All"]
    revenue = metric_by_group(orders, "revenue")["All"]
    count = metric_by_group(orders, "orders")["All"]
    assert (aov - revenue / count).abs().max() < 1e-9


@pytest.mark.parametrize("group", ["region", "category", "channel"])
def test_groups_add_back_up_to_the_total(orders, group):
    split = metric_by_group(orders, "revenue", group)
    total = metric_by_group(orders, "revenue")["All"]
    assert (split.sum(axis=1) - total).abs().max() < 1e-6
    assert set(split.columns) == set(orders[group].unique())


def test_months_run_oldest_first_and_are_labelled_as_periods(orders):
    index = list(metric_by_group(orders, "orders").index)
    assert index == sorted(index) and index[0] == "2025-09" and index[-1] == "2026-08"


def test_bad_metric_or_group_is_rejected(orders):
    with pytest.raises(ValueError):
        metric_by_group(orders, "profit")
    with pytest.raises(ValueError):
        metric_by_group(orders, "revenue", "customer_segment")


def test_baseline_index_puts_the_average_of_earlier_months_at_100(orders):
    idx = baseline_index(metric_by_group(orders, "orders")["All"])
    assert idx.iloc[:-1].mean() == pytest.approx(100.0)


# ---------------------------------------------------------- driver series ---

def test_marketing_trend_reproduces_the_findings_own_figures(orders, marketing, evidence):
    ev = evidence["stat_marketing_North"]
    t = driver_trend(ev, orders, marketing)
    assert t.segment == "North" and t.comparison == ev.details["control_segments"]
    assert last_over_prior(t.driver_segment) == pytest.approx(ev.details["driver_change_pct"], abs=0.3)
    assert last_over_prior(t.orders_segment) == pytest.approx(ev.details["raw_orders_change_pct"], abs=0.3)
    assert last_over_prior(t.orders_comparison) == pytest.approx(ev.details["control_orders_change_pct"], abs=0.3)
    assert len(t.months) == len(t.driver_segment) == len(t.orders_comparison) == 12


def test_price_trend_reproduces_the_like_for_like_price_change(orders, marketing, evidence):
    ev = evidence["stat_price_Electronics"]
    t = driver_trend(ev, orders, marketing)
    assert t.driver_label == "Like-for-like price"
    assert last_over_prior(t.driver_segment) == pytest.approx(ev.details["driver_change_pct"], abs=0.3)
    # Comparison categories kept typical prices, so their index sits flat.
    assert max(t.driver_comparison) - min(t.driver_comparison) < 1.0


def test_indexed_series_are_centred_on_100_before_the_latest_month(orders, marketing, evidence):
    t = driver_trend(evidence["stat_marketing_North"], orders, marketing)
    for series in (t.driver_segment, t.driver_comparison, t.orders_segment, t.orders_comparison):
        assert sum(series[:-1]) / len(series[:-1]) == pytest.approx(100.0, abs=0.2)


def test_no_trend_for_evidence_that_is_not_a_marketing_or_price_test(orders, marketing, evidence):
    assert driver_trend(evidence["obs_baseline_trend"], orders, marketing) is None
    assert driver_trend(evidence["assoc_region_North"], orders, marketing) is None
    assert driver_trend(evidence["chan_North_Paid"], orders, marketing) is None


def test_no_trend_when_the_finding_had_no_comparison_group(orders, marketing, evidence):
    ev = evidence["stat_marketing_North"]
    bare = type(ev)(**{**ev.__dict__, "details": {**ev.details, "control_segments": []}})
    assert driver_trend(bare, orders, marketing) is None


# ----------------------------------------------------------------- charts ---

def test_metric_figure_has_a_line_per_group_and_marks_the_latest_month(orders):
    frame = metric_by_group(orders, "revenue", "region")
    fig = metric_figure(frame, "Revenue by month, by region", money=True)
    lines = [t for t in fig.data if t.mode == "lines+markers"]
    assert [t.name for t in lines] == list(frame.columns)
    assert len(fig.data) == len(frame.columns) + 1          # plus the latest-month ring
    assert fig.data[-1].x[0] == frame.index[-1]


def test_driver_figure_draws_driver_and_orders_against_the_comparison(orders, marketing, evidence):
    ev = evidence["stat_marketing_North"]
    fig = driver_figure(driver_trend(ev, orders, marketing), ev.strength)
    assert len(fig.data) == 4
    names = [t.name for t in fig.data]
    assert names.count("North") == 2 and sum(n.startswith("comparison:") for n in names) == 2
    north = [t for t in fig.data if t.name == "North"]
    assert north[0].line.color == "#0f766e"                  # strong
