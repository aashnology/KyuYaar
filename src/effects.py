"""
Cause-testing tools -- Layer 2.

segment_breakdown() says WHERE the drop is concentrated. The two tools here
test WHY, one candidate cause each:

  marketing_effect()  -- did a region's marketing spend move, and did that
                         region's order volume move against comparable
                         segments in the same direction?
  price_effect()      -- did a category's like-for-like price move, and did
                         its order volume move the opposite way?

Both return "statistical" Evidence: an effect size with a significance test,
measured against a control group. They do not establish causation. A spend
cut and an order drop landing in the same month is consistent with the cause
but cannot rule out something else that changed at the same time, and every
returned object says so.

Method, in plain terms
----------------------
Order counts are compared between the latest period and the one before it.
A segment under test (say region=North) is compared only against segments
whose driver did NOT move -- regions with typical spend, categories with
typical prices. Comparing against "everyone else" would be wrong, because
the segment that really changed would sit inside its own control group and
make the untouched segments look like they had moved the other way.

The comparison is made *inside each stratum* of the other dimension (for a
region, each product category), so a category-wide effect such as a price hike
lands on the segment and its controls alike and cancels out instead of being
mistaken for a regional effect.

Within a stratum the comparison is a difference of log rate ratios. Counts are
Poisson-like, so the variance of a log ratio is roughly 1/n_last + 1/n_prior
for each group. Strata are pooled with inverse-variance weights, which yields
one adjusted effect, a standard error, a z-score and a two-sided p-value.

Insufficient data
-----------------
Some comparisons cannot be made from the data at hand: a spend change against
a month with no spend, a category with no product sold in both months, a
comparison group with no orders. Those cases do not raise. The tool returns an
Evidence object with strength "weak" (so it can never support a decision), no
value, and details["insufficient_data"] set, with a plain-language reason. That
is different from a tested result that came out weak: nothing was tested. The
thresholds for strong and moderate are not touched by any of this.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from evidence import Evidence, _downgrade_if_small_sample

# Thresholds on the adjusted order-volume effect (percent) and its p-value.
STRONG_EFFECT, STRONG_P = 15.0, 0.01
MODERATE_EFFECT, MODERATE_P = 8.0, 0.05

# How big the driver itself must move before it can be called a candidate.
MATERIAL_SPEND_CHANGE = 10.0   # percent, vs. spend in the comparison regions
MATERIAL_PRICE_CHANGE = 3.0    # percent, vs. prices in the comparison categories

_CONTINUITY = 0.5              # keeps empty cells from producing log(0)
_EPS = 1e-9                    # a spend or price at or below this counts as zero

_TIMING_CAVEAT = (
    "the driver and the order change occurred in the same period, so this "
    "cannot rule out another change made at the same time"
)


class InsufficientData(Exception):
    """A comparison that cannot be computed from the data at hand.

    Raised only inside this module and channels.py, and always caught there:
    each public tool turns it into an insufficient-data Evidence, so it never
    reaches a caller. `code` is a stable machine-readable reason; `reason` is
    a sentence a person can read.
    """

    def __init__(self, code, reason):
        super().__init__(reason)
        self.code = code
        self.reason = reason


def _insufficient_caveats(code, reason):
    """Caveat list for an insufficient-data result. The no-comparison-group
    wording predates this and is kept exactly as it was."""
    if code == "no_comparison_group":
        return [reason]
    return [
        f"insufficient data: {reason}",
        "this is a limit of the data, not a finding that the cause is absent",
    ]


def _insufficient_details(code, reason):
    return {"insufficient_data": True, "insufficient_reason": code, "insufficient_reason_text": reason}


def _undefined_change_reason(subject, prior, last, prior_label, last_label):
    """Why a percent change in `subject` cannot be computed, or None if it can."""
    prior_zero, last_zero = abs(prior) < _EPS, abs(last) < _EPS
    if prior_zero and last_zero:
        return ("no_driver_in_either_period",
                f"{subject} was zero in both {prior_label} and {last_label}, so there is no change to test")
    if prior_zero:
        return ("no_driver_baseline",
                f"{subject} was zero in {prior_label}, so a percent change against it is undefined")
    return None


def _last_two_periods(df, date_col="order_date", freq="M"):
    period = pd.to_datetime(df[date_col]).dt.to_period(freq)
    periods = sorted(period.unique())
    if len(periods) < 2:
        raise ValueError("need at least 2 periods to compare")
    return period, periods[-1], periods[-2]


def _adjusted_order_effect(df, in_segment, in_control, strata_col, period, last, prior):
    """Pooled log-rate-ratio difference between a segment and its controls,
    computed within each stratum. Returns (effect_pct, ci_low_pct, ci_high_pct,
    z, p_value, raw_segment_pct, raw_control_pct)."""
    seg_last = int((in_segment & (period == last)).sum())
    seg_prior = int((in_segment & (period == prior)).sum())
    ctl_last = int((in_control & (period == last)).sum())
    ctl_prior = int((in_control & (period == prior)).sum())
    # With no orders at all on one side, the log ratio would be built from the
    # continuity constant alone: a number, but not an estimate of anything.
    if seg_last + seg_prior == 0:
        raise InsufficientData("no_orders_in_segment",
                               "the segment recorded no orders in either compared month")
    if ctl_last + ctl_prior == 0:
        raise InsufficientData("no_orders_in_comparison_group",
                               "the comparison group recorded no orders in either compared month")
    strata = list(df.groupby(strata_col).groups.items())
    if not strata:
        raise InsufficientData("no_strata", f"there are no {strata_col} values to compare within")

    num, den = [], []
    for _, idx in strata:
        seg = in_segment.loc[idx]
        ctl = in_control.loc[idx]
        per = period.loc[idx]
        a1 = int((seg & (per == last)).sum()) + _CONTINUITY
        a0 = int((seg & (per == prior)).sum()) + _CONTINUITY
        c1 = int((ctl & (per == last)).sum()) + _CONTINUITY
        c0 = int((ctl & (per == prior)).sum()) + _CONTINUITY
        diff = math.log(a1 / a0) - math.log(c1 / c0)
        var = 1 / a1 + 1 / a0 + 1 / c1 + 1 / c0
        num.append(diff / var)
        den.append(1 / var)

    pooled = sum(num) / sum(den)
    se = math.sqrt(1 / sum(den))
    z = pooled / se
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    def to_pct(x):
        return (math.exp(x) - 1) * 100

    raw_seg = (seg_last / seg_prior - 1) * 100 if seg_prior else float("nan")
    raw_ctl = (ctl_last / ctl_prior - 1) * 100 if ctl_prior else float("nan")

    return (
        to_pct(pooled), to_pct(pooled - 1.96 * se), to_pct(pooled + 1.96 * se),
        z, p_value, raw_seg, raw_ctl,
    )


def _pick_controls(changes, target, material):
    """Given each segment's driver change, return (typical_change, moved,
    control_segments). Typical is the median across segments; a segment
    counts as having moved if it sits `material` points or more from it."""
    valid = {k: v for k, v in changes.items() if math.isfinite(v)}
    typical = float(np.median(list(valid.values()))) if valid else 0.0
    own = valid.get(target, float("nan"))
    moved = math.isfinite(own) and abs(own - typical) >= material
    controls = [k for k, v in valid.items() if k != target and abs(v - typical) < material]
    return typical, moved, controls


def _grade(driver_ok, effect_ok_sign, effect_pct, p_value):
    """Map (was there a real driver move, did volume move the expected way,
    how big, how certain) onto the Layer 1 strength scale."""
    if not driver_ok or not effect_ok_sign:
        return "weak"
    size = abs(effect_pct)
    if size >= STRONG_EFFECT and p_value < STRONG_P:
        return "strong"
    if size >= MODERATE_EFFECT and p_value < MODERATE_P:
        return "moderate"
    return "weak"


def _signed(x):
    """Format a percentage with an explicit sign, avoiding '-0%'."""
    if not math.isfinite(x):
        return "n/a"
    r = round(x)
    return f"{r:+d}%" if r else "+0%"


def _p_text(p_value):
    """Readable p-value that also survives the numeric guardrail."""
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def _pct(new, old):
    """Percent change, or NaN when it is undefined (a zero or missing baseline)."""
    if not (math.isfinite(new) and math.isfinite(old)) or abs(old) < _EPS:
        return float("nan")
    return (new / old - 1) * 100


def _revenue_at_stake(prior_revenue, net_revenue_effect_pct):
    """Estimated monthly revenue lost relative to comparable segments.
    Zero when the net effect is not a loss."""
    if net_revenue_effect_pct >= 0 or math.isnan(net_revenue_effect_pct):
        return 0.0
    return round(prior_revenue * abs(net_revenue_effect_pct) / 100, 2)


def _sorted(results):
    order = {"strong": 0, "moderate": 1, "weak": 2}
    return sorted(
        results,
        key=lambda e: (order[e.strength], -(abs(e.value) if e.value is not None else 0)),
    )


def marketing_effect(orders, marketing, region_col="region"):
    """
    Test "a change in marketing spend explains the order-volume change" for
    each region. One Evidence object per region, strongest first.

    Spend change is total spend across channels for the region, latest period
    vs. the prior one. The largest single-channel move is named in the
    hypothesis so the reader can see what actually changed. A region is only
    a candidate if its spend moved away from the typical regional change;
    its orders are then compared with regions whose spend did not.

    A region whose test cannot be run (no marketing rows for one of the two
    months, no spend to measure a change from, no comparison group, no orders)
    gets an insufficient-data Evidence instead of an error.
    """
    period, last, prior = _last_two_periods(orders)

    mkt = marketing.copy()
    mkt["_period"] = pd.to_datetime(mkt["date"]).dt.to_period("M")
    mkt = mkt[mkt["_period"].isin([last, prior])]

    # A marketing table that does not reach one of the compared months says
    # nothing about spend in it. Absent rows are not the same as zero spend.
    uncovered = [str(p) for p in (prior, last) if not (mkt["_period"] == p).any()]
    coverage_gap = None
    if uncovered:
        coverage_gap = (
            "no_marketing_data",
            f"the marketing table has no rows for {' and '.join(uncovered)}, so spend "
            f"cannot be compared across the two months",
        )

    regions = sorted(orders[region_col].dropna().unique())
    spend = {
        r: (
            float(mkt.loc[(mkt["region"] == r) & (mkt["_period"] == prior), "spend"].sum()),
            float(mkt.loc[(mkt["region"] == r) & (mkt["_period"] == last), "spend"].sum()),
        )
        for r in regions
    }
    spend_change = {r: _pct(v[1], v[0]) for r, v in spend.items()}

    results = []
    for region in regions:
        typical, moved, controls = _pick_controls(spend_change, region, MATERIAL_SPEND_CHANGE)
        in_region = orders[region_col] == region
        in_control = orders[region_col].isin(controls)
        seg_orders_last = int((in_region & (period == last)).sum())

        own_change = spend_change[region]
        spend_prior, spend_last = spend[region]

        by_channel = (
            mkt[mkt["region"] == region]
            .pivot_table(index="channel", columns="_period", values="spend", aggfunc="sum")
            .reindex(columns=[prior, last])
            .fillna(0)
        )
        channel_delta = by_channel[last] - by_channel[prior]
        top_channel = channel_delta.abs().idxmax() if len(channel_delta) else None

        rev_prior = float(orders.loc[in_region & (period == prior), "revenue"].sum())
        rev_last = float(orders.loc[in_region & (period == last), "revenue"].sum())

        insufficient = coverage_gap
        if insufficient is None:
            insufficient = _undefined_change_reason(
                f"marketing spend in {region}", spend_prior, spend_last, str(prior), str(last)
            )
        if insufficient is None and not controls:
            # Every region moved, so there is nothing unchanged to compare with.
            insufficient = (
                "no_comparison_group",
                "no region kept typical spend, so there is no comparison group for this region",
            )
        if insufficient is None:
            try:
                (effect, ci_lo, ci_hi, z, p_value, raw_seg, raw_ctl) = _adjusted_order_effect(
                    orders, in_region, in_control, "category", period, last, prior
                )
            except InsufficientData as exc:
                insufficient = (exc.code, exc.reason)

        caveats = []
        if insufficient is not None:
            effect = ci_lo = ci_hi = z = raw_seg = raw_ctl = float("nan")
            p_value = 1.0
            strength = "weak"
            caveats.extend(_insufficient_caveats(*insufficient))
        else:
            gap = own_change - typical
            sign_ok = moved and ((gap < 0 and effect < 0) or (gap > 0 and effect > 0))
            strength = _grade(moved, sign_ok, effect, p_value)
            strength, small = _downgrade_if_small_sample(strength, seg_orders_last)
            caveats.extend(small)

            if not moved:
                caveats.append(
                    "spend in this region stayed close to the typical regional "
                    "change, so marketing is not a candidate explanation here"
                )
            elif not sign_ok:
                caveats.append(
                    "orders moved in the opposite direction to what the spend "
                    "change would predict"
                )
            else:
                caveats.append(_TIMING_CAVEAT)

        has_fit = insufficient is None
        candidate = moved and strength != "weak"
        if insufficient is not None and insufficient[0] == "no_comparison_group":
            hypothesis = (
                f"In {region}, marketing spend moved {_signed(own_change)}, but no "
                f"region kept typical spend to compare orders against"
            )
        elif insufficient is not None:
            hypothesis = (
                f"In {region}, marketing spend cannot be tested as an explanation "
                f"(insufficient data: {insufficient[1]})"
            )
        elif moved:
            kind = "cut" if own_change < typical else "increase"
            hypothesis = (
                f"In {region}, a {kind} in marketing spend ({_signed(own_change)} vs. "
                f"{_signed(typical)} typical elsewhere"
                + (f", mainly {top_channel}" if top_channel else "")
                + f") accompanies an order-volume change of {_signed(effect)} "
                f"relative to regions with typical spend"
            )
        else:
            hypothesis = (
                f"In {region}, marketing spend stayed in line with other regions "
                f"({_signed(own_change)} vs. {_signed(typical)}); order volume changed "
                f"{_signed(effect)} relative to regions with typical spend"
            )

        details = {
            "period": str(last),
            "prior_period": str(prior),
            "driver": "marketing_spend",
            "driver_moved": bool(moved),
            "driver_change_pct": None if math.isnan(own_change) else round(own_change, 1),
            "driver_typical_change_pct": round(typical, 1),
            "control_segments": controls,
            "top_channel": top_channel if (has_fit or insufficient[0] == "no_comparison_group") else None,
            "raw_orders_change_pct": round(raw_seg, 1) if has_fit else None,
            "control_orders_change_pct": round(raw_ctl, 1) if has_fit else None,
            "ci_low_pct": round(ci_lo, 1) if has_fit else None,
            "ci_high_pct": round(ci_hi, 1) if has_fit else None,
            "z": round(z, 2) if has_fit else None,
            "p_value": float(f"{p_value:.3g}") if has_fit else None,
            "p_value_text": _p_text(p_value) if has_fit else None,
            "spend_prior": round(spend_prior, 2),
            "spend_last": round(spend_last, 2),
            "revenue_prior": round(rev_prior, 2),
            "revenue_last": round(rev_last, 2),
            # Assumes average order value is unchanged, so an order-volume
            # shortfall maps one-for-one onto revenue.
            "revenue_at_stake": _revenue_at_stake(rev_prior, effect) if candidate else None,
        }
        if insufficient is not None:
            details.update(_insufficient_details(*insufficient))

        results.append(Evidence(
            id=f"stat_marketing_{region}",
            hypothesis=hypothesis,
            evidence_type="statistical",
            metric="orders_change_vs_comparable_segments_pct",
            value=round(effect, 1) if has_fit else None,
            baseline=0.0,
            segment=f"{region_col}={region}",
            strength=strength,
            sample_size=seg_orders_last,
            caveats=caveats,
            details=details,
        ))

    return _sorted(results)


def _like_for_like_price_change(orders, category_col, period, last, prior):
    """Revenue-weighted change in each product's average unit price between
    the two periods, per category. Only products sold in both periods count,
    so a shift in product mix does not masquerade as a price move. A category
    with no such product is left out of the result: its price change is
    undefined, and price_effect reports it as insufficient data."""
    scoped = orders[period.isin([last, prior])].assign(_period=period)
    price = scoped.pivot_table(
        index=["_period", category_col, "product_id"],
        values="unit_price", aggfunc="mean",
    ).reset_index()
    wide = (
        price.pivot_table(
            index=[category_col, "product_id"], columns="_period", values="unit_price"
        )
        .reindex(columns=[prior, last])
        .dropna()
    )
    # A prior price of zero has no percent change; leave those products out.
    wide = wide[wide[prior] > _EPS]
    weights = (
        scoped[scoped["_period"] == prior]
        .groupby(["product_id"])["revenue"].sum()
    )
    wide = wide.join(weights.rename("w"), on="product_id")
    wide["chg"] = (wide[last] / wide[prior] - 1) * 100
    out = {}
    for cat, grp in wide.groupby(level=0):
        usable = grp[np.isfinite(grp["chg"]) & (grp["w"] > 0)]
        if len(usable):
            out[cat] = float(np.average(usable["chg"], weights=usable["w"]))
    return out


def price_effect(orders, category_col="category"):
    """
    Test "a price change explains the order-volume change" for each product
    category. One Evidence object per category, strongest first.

    Price change is like-for-like (same products in both periods). A category
    is only a candidate if its price moved away from the typical category
    change; its orders are then compared with categories whose prices did not
    move, adjusted for regional mix.

    A category whose test cannot be run (no product sold in both months, no
    comparison group, no orders) gets an insufficient-data Evidence instead of
    an error.
    """
    period, last, prior = _last_two_periods(orders)
    price_change = _like_for_like_price_change(orders, category_col, period, last, prior)

    results = []
    for category in sorted(orders[category_col].dropna().unique()):
        typical, moved, controls = _pick_controls(price_change, category, MATERIAL_PRICE_CHANGE)
        in_cat = orders[category_col] == category
        in_control = orders[category_col].isin(controls)
        seg_orders_last = int((in_cat & (period == last)).sum())

        own_change = price_change.get(category, float("nan"))
        rev_prior = float(orders.loc[in_cat & (period == prior), "revenue"].sum())
        rev_last = float(orders.loc[in_cat & (period == last), "revenue"].sum())

        insufficient = None
        if not math.isfinite(own_change):
            insufficient = (
                "no_price_baseline",
                f"no product in {category} was sold in both {prior} and {last} at a "
                f"non-zero price, so a like-for-like price change cannot be measured",
            )
        elif not controls:
            insufficient = (
                "no_comparison_group",
                "no category kept typical prices, so there is no comparison "
                "group for this category",
            )
        else:
            try:
                (effect, ci_lo, ci_hi, z, p_value, raw_seg, raw_ctl) = _adjusted_order_effect(
                    orders, in_cat, in_control, "region", period, last, prior
                )
            except InsufficientData as exc:
                insufficient = (exc.code, exc.reason)

        caveats = []
        net_revenue_effect = float("nan")
        if insufficient is not None:
            effect = ci_lo = ci_hi = z = raw_seg = raw_ctl = float("nan")
            p_value = 1.0
            strength = "weak"
            caveats.extend(_insufficient_caveats(*insufficient))
        else:
            gap = own_change - typical
            # A price rise should reduce volume; a price cut should raise it.
            sign_ok = moved and ((gap > 0 and effect < 0) or (gap < 0 and effect > 0))
            strength = _grade(moved, sign_ok, effect, p_value)
            strength, small = _downgrade_if_small_sample(strength, seg_orders_last)
            caveats.extend(small)

            if not moved:
                caveats.append(
                    "prices in this category stayed close to the typical "
                    "category change, so pricing is not a candidate explanation here"
                )
            elif not sign_ok:
                caveats.append(
                    "orders moved in the opposite direction to what the price "
                    "change would predict"
                )
            else:
                caveats.append(_TIMING_CAVEAT)
                caveats.append(
                    "the price effect is read from a single period; the size of "
                    "the demand response may not persist"
                )
            # Higher price per order times fewer orders, relative to controls.
            net_revenue_effect = ((1 + gap / 100) * (1 + effect / 100) - 1) * 100

        has_fit = insufficient is None
        candidate = moved and strength != "weak"
        if insufficient is not None and insufficient[0] == "no_comparison_group":
            hypothesis = (
                f"In {category}, prices moved {_signed(own_change)}, but no category "
                f"kept typical prices to compare orders against"
            )
        elif insufficient is not None:
            hypothesis = (
                f"In {category}, pricing cannot be tested as an explanation "
                f"(insufficient data: {insufficient[1]})"
            )
        elif moved:
            kind = "increase" if own_change > typical else "decrease"
            hypothesis = (
                f"In {category}, a like-for-like price {kind} ({_signed(own_change)} vs. "
                f"{_signed(typical)} typical elsewhere) accompanies an order-volume "
                f"change of {_signed(effect)} relative to categories with typical prices"
            )
        else:
            hypothesis = (
                f"In {category}, prices stayed in line with other categories "
                f"({_signed(own_change)} vs. {_signed(typical)}); order volume changed "
                f"{_signed(effect)} relative to categories with typical prices"
            )

        details = {
            "period": str(last),
            "prior_period": str(prior),
            "driver": "unit_price",
            "driver_moved": bool(moved),
            "driver_change_pct": None if math.isnan(own_change) else round(own_change, 1),
            "driver_typical_change_pct": round(typical, 1),
            "control_segments": controls,
            "raw_orders_change_pct": round(raw_seg, 1) if has_fit else None,
            "control_orders_change_pct": round(raw_ctl, 1) if has_fit else None,
            "ci_low_pct": round(ci_lo, 1) if has_fit else None,
            "ci_high_pct": round(ci_hi, 1) if has_fit else None,
            "z": round(z, 2) if has_fit else None,
            "p_value": float(f"{p_value:.3g}") if has_fit else None,
            "p_value_text": _p_text(p_value) if has_fit else None,
            "revenue_prior": round(rev_prior, 2),
            "revenue_last": round(rev_last, 2),
            "net_revenue_effect_pct": None if math.isnan(net_revenue_effect) else round(net_revenue_effect, 1),
            "revenue_at_stake": _revenue_at_stake(rev_prior, net_revenue_effect) if candidate else None,
        }
        if insufficient is not None:
            details.update(_insufficient_details(*insufficient))

        results.append(Evidence(
            id=f"stat_price_{category}",
            hypothesis=hypothesis,
            evidence_type="statistical",
            metric="orders_change_vs_comparable_segments_pct",
            value=round(effect, 1) if has_fit else None,
            baseline=0.0,
            segment=f"{category_col}={category}",
            strength=strength,
            sample_size=seg_orders_last,
            caveats=caveats,
            details=details,
        ))

    return _sorted(results)
