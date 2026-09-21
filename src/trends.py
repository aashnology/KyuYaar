"""
Trend series behind the evidence charts -- Layer 6.

Two kinds of series, both plain pandas over the order and marketing tables:

  metric_by_group()  monthly revenue, orders or average order value, overall
                     or split by region / category / channel.
  driver_trend()     for one supported cause, the driver (marketing spend or
                     like-for-like price) and the order count of the segment
                     under test, each next to the comparison segments the
                     finding was measured against.

Nothing here is modelled. The charts show the same tables the findings were
computed from, so a reader can see the move the finding describes instead of
taking a percentage on trust.

Driver and order series are indexed so the average of all earlier months is
100. A finding that says "spend fell 34% against a typical 1%" then shows up as
the last point sitting well under 100 for the segment and near 100 for its
comparison group.
"""

from dataclasses import dataclass, field

import pandas as pd

from effects import _like_for_like_price_change

METRICS = {"revenue": "Revenue", "orders": "Orders", "aov": "Average order value"}
GROUPS = {"region": "Region", "category": "Category", "channel": "Channel"}


def _months(orders):
    return pd.to_datetime(orders["order_date"]).dt.to_period("M")


def _label(period):
    return str(period)


def metric_by_group(orders, metric="revenue", group_col=None) -> pd.DataFrame:
    """Monthly `metric`, one column per group (a single column, 'All', when
    `group_col` is None). Rows are months, oldest first."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}, got {metric!r}")
    if group_col is not None and group_col not in GROUPS:
        raise ValueError(f"group_col must be one of {list(GROUPS)} or None, got {group_col!r}")

    work = orders.assign(_month=_months(orders))
    work["_group"] = "All" if group_col is None else work[group_col]
    grouped = work.groupby(["_month", "_group"])
    revenue = grouped["revenue"].sum().unstack(fill_value=0.0)
    count = grouped["revenue"].size().unstack(fill_value=0)

    if metric == "revenue":
        out = revenue
    elif metric == "orders":
        out = count
    else:
        out = revenue / count.where(count > 0)      # no orders -> no average, not zero
    out = out.astype(float)
    out.index = [_label(p) for p in out.index]
    out.columns.name = None
    return out


def baseline_index(series) -> pd.Series:
    """Rescale so the average of every month except the last is 100."""
    base = series.iloc[:-1].mean()
    if not base or pd.isna(base):
        return series * float("nan")
    return series / base * 100


@dataclass
class DriverTrend:
    evidence_id: str
    segment: str                      # 'North', 'Electronics'
    driver: str                       # 'marketing_spend' | 'unit_price'
    driver_label: str
    months: list
    driver_segment: list
    driver_comparison: list
    orders_segment: list
    orders_comparison: list
    comparison: list = field(default_factory=list)   # names of the comparison segments


def _spend_by_month(marketing, region_col, names, months):
    mkt = marketing.assign(_month=pd.to_datetime(marketing["date"]).dt.to_period("M"))
    scoped = mkt[mkt[region_col].isin(names)]
    spend = scoped.groupby("_month")["spend"].sum()
    return spend.reindex(months)


def _orders_by_month(orders, col, names, months):
    scoped = orders[orders[col].isin(names)]
    return scoped.groupby(_months(scoped)).size().reindex(months, fill_value=0).astype(float)


def _price_index(orders, col, names, months):
    """Chain-linked like-for-like price level: each month's change is measured
    on products sold in both that month and the one before, exactly as the
    price test measures it. A month with no comparable change carries the
    previous level forward."""
    period = _months(orders)
    levels = {}
    for name in names:
        level, series = 100.0, [100.0]
        for prior, last in zip(months[:-1], months[1:]):
            change = _like_for_like_price_change(orders, col, period, last, prior).get(name)
            if change is not None and not pd.isna(change):
                level *= 1 + change / 100
            series.append(level)
        levels[name] = pd.Series(series, index=months)
    return pd.DataFrame(levels).mean(axis=1)


def driver_trend(ev, orders, marketing):
    """The driver and order series behind one statistical finding, or None when
    the finding has no comparison group or is not a marketing or price test."""
    if ev.evidence_type != "statistical" or not ev.segment or "=" not in ev.segment:
        return None
    controls = list(ev.details.get("control_segments") or [])
    driver = ev.details.get("driver")
    if not controls or driver not in ("marketing_spend", "unit_price"):
        return None

    col, name = ev.segment.split("=", 1)
    months = sorted(_months(orders).unique())
    if len(months) < 3:
        return None

    if driver == "marketing_spend":
        seg_driver = _spend_by_month(marketing, col, [name], months)
        ctl_driver = _spend_by_month(marketing, col, controls, months)
        label = "Marketing spend"
    else:
        seg_driver = _price_index(orders, col, [name], months)
        ctl_driver = _price_index(orders, col, controls, months)
        label = "Like-for-like price"

    seg_orders = _orders_by_month(orders, col, [name], months)
    ctl_orders = _orders_by_month(orders, col, controls, months)

    def out(series):
        return [None if pd.isna(v) else round(float(v), 1) for v in baseline_index(series)]

    return DriverTrend(
        evidence_id=ev.id, segment=name, driver=driver, driver_label=label,
        months=[_label(m) for m in months],
        driver_segment=out(seg_driver), driver_comparison=out(ctl_driver),
        orders_segment=out(seg_orders), orders_comparison=out(ctl_orders),
        comparison=controls,
    )
