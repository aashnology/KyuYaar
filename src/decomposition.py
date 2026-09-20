"""
Orders vs. order value -- Layer 3.

Revenue is orders x average order value (AOV), so any change in revenue comes
from fewer/more orders, smaller/larger orders, or both. baseline_trend() says
that revenue moved and segment_breakdown() says where; this module says which
of those two parts moved, so that a candidate cause can be checked against the
pattern it predicts. A marketing cut should cost orders and leave order size
alone. A price rise should cost orders and lift order value.

Two hypotheses are returned for each group, each with its own strength:

    "the number of orders changed"        (metric: orders_change_pct)
    "average order value changed"         (metric: aov_change_pct)

Splitting the revenue change
----------------------------
With N orders, average order value A and revenue R = N * A, comparing the
latest month (1) with the one before it (0):

    volume effect = (N1 - N0) * (A0 + A1) / 2
    AOV effect    = (A1 - A0) * (N0 + N1) / 2

The two add up to R1 - R0 exactly. Each is the change in one factor priced at
the average of the two months' value of the other factor, which is the mean of
the two possible orderings, so neither factor is favoured and there is no
leftover interaction term. Effects are reported in points of prior-month
revenue, so they add up to the revenue change in percent.

Is the move real?
-----------------
Monthly order counts and AOV both wander. A raw two-month split has no control
group to cancel that noise, so each factor's latest month-on-month change is
compared with how much the same figure changed month on month in all earlier
months, using a prediction-interval t statistic. A move that is ordinary for
this business grades weak however large it looks. With too little history the
grade is weak and says so.

Strength uses the same size and p-value bars as the Layer 2 cause tests. It
rates how clearly the data shows that the factor moved. It does not say why,
and it does not say how much the factor matters to the total, which is what
the points-of-revenue figures are for.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from effects import _grade, _p_text
from evidence import Evidence, _downgrade_if_small_sample

# Earlier month-on-month changes needed before "normal variation" means anything.
MIN_HISTORY = 6

_STRENGTH_ORDER = {"strong": 0, "moderate": 1, "weak": 2}

_NOT_A_CAUSE = "this describes how revenue changed, not why it changed"
_MIX_CAVEAT = (
    "average order value blends different products and order sizes, so a change "
    "can reflect a shift in what is bought rather than a change in how much each "
    "order is worth"
)


def _monthly(df, mask, period, all_periods):
    """Orders, revenue and AOV per month for one group, on a gap-free calendar."""
    sub = df.loc[mask]
    grouped = sub.groupby(period.loc[mask]).agg(n=("revenue", "size"), rev=("revenue", "sum"))
    grouped = grouped.reindex(all_periods, fill_value=0)
    grouped["aov"] = np.where(grouped["n"] > 0, grouped["rev"] / grouped["n"].replace(0, np.nan), np.nan)
    return grouped


def _change_test(series):
    """Latest month-on-month % change of `series` and how unusual it is against
    the earlier month-on-month changes. Returns None if the latest change cannot
    be computed; p_value is None when there is too little history to judge."""
    values = series.to_numpy(dtype=float)
    prev, last = values[-2], values[-1]
    if not (prev > 0 and last > 0):
        return None
    latest = (last / prev - 1) * 100

    with np.errstate(divide="ignore", invalid="ignore"):
        changes = (values[1:-1] / values[:-2] - 1) * 100
    hist = changes[np.isfinite(changes)]

    out = {"latest_pct": latest, "history_n": int(len(hist)),
           "history_mean_pct": None, "history_std_pct": None, "t": None, "p_value": None}
    if len(hist) < MIN_HISTORY:
        return out

    mean, std = float(hist.mean()), float(hist.std(ddof=1))
    out["history_mean_pct"], out["history_std_pct"] = mean, std
    if std > 0:
        t = (latest - mean) / (std * math.sqrt(1 + 1 / len(hist)))
        out["t"] = t
        out["p_value"] = float(2 * stats.t.sf(abs(t), df=len(hist) - 1))
    else:
        out["p_value"] = 1.0
    return out


def _grade_factor(test, min_orders):
    """Strength for one factor, plus the caveats that explain it."""
    caveats = []
    if test["p_value"] is None:
        caveats.append(
            f"only {test['history_n']} earlier month-on-month changes are available; "
            f"at least {MIN_HISTORY} are needed to judge what is normal for this business"
        )
        return "weak", caveats
    strength = _grade(True, True, test["latest_pct"], test["p_value"])
    strength, small = _downgrade_if_small_sample(strength, min_orders)
    caveats.extend(small)
    if strength == "weak" and not small:
        caveats.append(
            "this change is within the range this figure normally moves from one "
            "month to the next, so it is not distinguishable from ordinary variation"
        )
    return strength, caveats


def _fmt_orders(name, d):
    direction = "fell" if d["orders_change_pct"] < 0 else "rose"
    return (
        f"{name} order count {direction} {abs(d['orders_change_pct']):.1f}% vs. the prior "
        f"month ({d['orders_prior']:,} to {d['orders_last']:,} orders), worth "
        f"{d['orders_effect_pts']:+.1f} points of the {d['revenue_change_pct']:+.1f}% revenue change"
    )


def _fmt_aov(name, d):
    direction = "fell" if d["aov_change_pct"] < 0 else "rose"
    return (
        f"{name} average order value {direction} {abs(d['aov_change_pct']):.1f}% vs. the prior "
        f"month ({d['aov_prior']:,.2f} to {d['aov_last']:,.2f}), worth "
        f"{d['aov_effect_pts']:+.1f} points of the {d['revenue_change_pct']:+.1f}% revenue change"
    )


def _decompose_group(df, mask, period, all_periods, last, prior, dimension_col, value):
    """The two Evidence objects (orders, AOV) for one group of orders."""
    m = _monthly(df, mask, period, all_periods)
    n0, n1 = int(m["n"].iloc[-2]), int(m["n"].iloc[-1])
    r0, r1 = float(m["rev"].iloc[-2]), float(m["rev"].iloc[-1])
    segment = None if dimension_col is None else f"{dimension_col}={value}"
    tag = "overall" if dimension_col is None else f"{dimension_col}_{value}"
    name = "Overall," if dimension_col is None else f"In {value},"
    period_info = {"period": str(last), "prior_period": str(prior)}

    t_orders = _change_test(m["n"])
    t_aov = _change_test(m["aov"])

    if n0 == 0 or n1 == 0 or t_orders is None or t_aov is None:
        # Nothing to compare: no orders (or no revenue) in one of the two months.
        ev = []
        for kind, metric in (("orders", "orders_change_pct"), ("aov", "aov_change_pct")):
            ev.append(Evidence(
                id=f"decomp_{kind}_{tag}",
                hypothesis=f"{name} the split between order count and order value cannot be computed",
                evidence_type="decomposition", metric=metric, value=None, baseline=None,
                segment=segment, strength="weak", sample_size=n1,
                caveats=["there were no orders, or no revenue, in one of the two months, so there is nothing to compare"],
                details={**period_info, "leading_factor": None},
            ))
        return ev

    a0, a1 = r0 / n0, r1 / n1
    orders_effect = (n1 - n0) * (a0 + a1) / 2
    aov_effect = (a1 - a0) * (n0 + n1) / 2
    orders_pts = orders_effect / r0 * 100
    aov_pts = aov_effect / r0 * 100
    revenue_pct = (r1 / r0 - 1) * 100

    min_orders = min(n0, n1)
    s_orders, c_orders = _grade_factor(t_orders, min_orders)
    s_aov, c_aov = _grade_factor(t_aov, min_orders)

    days0, days1 = prior.days_in_month, last.days_in_month
    if days0 != days1:
        c_orders.append(
            f"the two months have different lengths ({days0} vs. {days1} days), which by "
            f"itself shifts the number of orders"
        )

    # The leading factor is whichever supported move contributes more revenue.
    contenders = [(abs(orders_pts), "orders")] * (s_orders != "weak") + \
                 [(abs(aov_pts), "aov")] * (s_aov != "weak")
    leading = max(contenders)[1] if contenders else None

    shared = {
        **period_info,
        "orders_prior": n0, "orders_last": n1,
        "aov_prior": round(a0, 2), "aov_last": round(a1, 2),
        "revenue_prior": round(r0, 2), "revenue_last": round(r1, 2),
        "revenue_change_pct": round(revenue_pct, 1),
        "orders_change_pct": round(float(t_orders["latest_pct"]), 1),
        "aov_change_pct": round(float(t_aov["latest_pct"]), 1),
        "orders_effect": round(orders_effect, 2), "aov_effect": round(aov_effect, 2),
        "orders_effect_pts": round(orders_pts, 1), "aov_effect_pts": round(aov_pts, 1),
        "days_prior": days0, "days_last": days1,
        "leading_factor": leading,
    }

    def stats_block(t):
        return {
            "history_changes": t["history_n"],
            "history_mean_change_pct": None if t["history_mean_pct"] is None else round(t["history_mean_pct"], 1),
            "history_std_change_pct": None if t["history_std_pct"] is None else round(t["history_std_pct"], 1),
            "t": None if t["t"] is None else round(t["t"], 2),
            "p_value": None if t["p_value"] is None else float(f"{t['p_value']:.3g}"),
            "p_value_text": None if t["p_value"] is None else _p_text(t["p_value"]),
        }

    orders_ev = Evidence(
        id=f"decomp_orders_{tag}",
        hypothesis=_fmt_orders(name, shared),
        evidence_type="decomposition", metric="orders_change_pct",
        value=shared["orders_change_pct"], baseline=float(n0),
        segment=segment, strength=s_orders, sample_size=n1,
        caveats=c_orders + ([_NOT_A_CAUSE] if s_orders != "weak" else []),
        details={**shared, "factor": "orders", **stats_block(t_orders)},
    )
    aov_ev = Evidence(
        id=f"decomp_aov_{tag}",
        hypothesis=_fmt_aov(name, shared),
        evidence_type="decomposition", metric="aov_change_pct",
        value=shared["aov_change_pct"], baseline=round(a0, 2),
        segment=segment, strength=s_aov, sample_size=n1,
        caveats=c_aov + ([_MIX_CAVEAT, _NOT_A_CAUSE] if s_aov != "weak" else []),
        details={**shared, "factor": "aov", **stats_block(t_aov)},
    )
    return [orders_ev, aov_ev]


def aov_volume_decomposition(orders, dimension_col=None, date_col="order_date"):
    """
    Split the latest month's revenue change into an order-count part and an
    order-value part, overall or for each value of `dimension_col`.

    Returns two Evidence objects per group (orders, then AOV), type
    "decomposition". For a dimension the groups are ordered with the most
    clearly supported findings first.
    """
    period = pd.to_datetime(orders[date_col]).dt.to_period("M")
    if period.nunique() < 2:
        raise ValueError("need at least 2 periods to compare")
    all_periods = pd.period_range(period.min(), period.max(), freq="M")
    last, prior = all_periods[-1], all_periods[-2]

    if dimension_col is None:
        mask = pd.Series(True, index=orders.index)
        return _decompose_group(orders, mask, period, all_periods, last, prior, None, None)

    groups = []
    for value in sorted(orders[dimension_col].unique()):
        mask = orders[dimension_col] == value
        pair = _decompose_group(orders, mask, period, all_periods, last, prior, dimension_col, value)
        best = min(_STRENGTH_ORDER[e.strength] for e in pair)
        size = max(abs(e.value or 0) for e in pair)
        groups.append((best, -size, pair))
    groups.sort(key=lambda g: (g[0], g[1]))
    return [e for _, _, pair in groups for e in pair]


def _move(ev):
    return f"{'down' if ev.value < 0 else 'up'} {abs(ev.value):.1f}%"


def signature_check(cause, evidence):
    """
    Compare a supported cause with the order pattern it predicts.

      marketing spend cut  -> fewer orders, order value about unchanged
      price increase       -> fewer orders, higher order value
      (and the mirror image for a spend increase or a price cut)

    `cause` is a marketing or price Evidence object; `evidence` is everything
    gathered so far. Returns None if the matching decomposition was not run,
    otherwise {"status": "consistent" | "partial" | "inconsistent", "text": ...}.
    The text quotes figures straight from the decomposition Evidence.
    """
    if not cause.segment or "=" not in cause.segment:
        return None
    dim, value = cause.segment.split("=", 1)
    by_id = {e.id: e for e in evidence}
    orders = by_id.get(f"decomp_orders_{dim}_{value}")
    aov = by_id.get(f"decomp_aov_{dim}_{value}")
    if orders is None or aov is None or orders.value is None or aov.value is None:
        return None

    d = cause.details
    gap = (d.get("driver_change_pct") or 0) - (d.get("driver_typical_change_pct") or 0)
    o_move, a_move = orders.strength != "weak", aov.strength != "weak"
    o_phrase = f"order count {_move(orders)}"
    a_phrase = f"average order value {_move(aov)}"
    a_flat = f"average order value ({_move(aov)}) is not clearly different from ordinary variation"

    if d.get("driver") == "unit_price":
        orders_dir_ok = (orders.value < 0) == (gap > 0)
        aov_dir_ok = (aov.value > 0) == (gap > 0)
        expect = "fewer orders with a higher order value" if gap > 0 else "more orders with a lower order value"
        if o_move and orders_dir_ok and a_move and aov_dir_ok:
            status = "consistent"
            text = f"In {value}, {o_phrase} and {a_phrase}: the pattern a price change predicts ({expect})."
        elif o_move and orders_dir_ok and not a_move:
            status = "partial"
            text = (f"In {value}, {o_phrase}, but {a_flat}. A price change predicts {expect}, "
                    f"so only the order-count half is seen.")
        else:
            status = "inconsistent"
            text = (f"In {value}, {o_phrase} and {a_phrase}, which does not match the pattern "
                    f"a price change predicts ({expect}).")
    else:  # marketing spend
        orders_dir_ok = (orders.value < 0) == (gap < 0)
        if o_move and orders_dir_ok and not a_move:
            status = "consistent"
            text = (f"In {value}, {o_phrase}, while {a_flat}. That is the pattern a change in "
                    f"marketing spend predicts: order count moves, order size does not.")
        elif o_move and orders_dir_ok and a_move:
            status = "inconsistent"
            text = (f"In {value}, {o_phrase}, but {a_phrase} also moved, so an estimate that "
                    f"assumes unchanged order value is rougher than it looks.")
        else:
            status = "partial"
            text = (f"In {value}, order count on its own shows no clear move ({_move(orders)}); "
                    f"the marketing effect only appears against regions with typical spend.")
    return {"status": status, "text": text, "orders": orders, "aov": aov}
