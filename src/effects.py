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

_TIMING_CAVEAT = (
    "the driver and the order change occurred in the same period, so this "
    "cannot rule out another change made at the same time"
)


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
    num, den = [], []
    for _, idx in df.groupby(strata_col).groups.items():
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

    seg_last = int((in_segment & (period == last)).sum())
    seg_prior = int((in_segment & (period == prior)).sum())
    ctl_last = int((in_control & (period == last)).sum())
    ctl_prior = int((in_control & (period == prior)).sum())
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
    valid = {k: v for k, v in changes.items() if not math.isnan(v)}
    typical = float(np.median(list(valid.values()))) if valid else 0.0
    own = valid.get(target, float("nan"))
    moved = (not math.isnan(own)) and abs(own - typical) >= material
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
    r = round(x)
    return f"{r:+d}%" if r else "+0%"


def _p_text(p_value):
    """Readable p-value that also survives the numeric guardrail."""
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def _pct(new, old):
    return (new / old - 1) * 100 if old else float("nan")


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
    """
    period, last, prior = _last_two_periods(orders)

    mkt = marketing.copy()
    mkt["_period"] = pd.to_datetime(mkt["date"]).dt.to_period("M")
    mkt = mkt[mkt["_period"].isin([last, prior])]

    regions = sorted(orders[region_col].unique())
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
            .fillna(0)
        )
        channel_delta = by_channel[last] - by_channel[prior]
        top_channel = channel_delta.abs().idxmax() if len(channel_delta) else None

        rev_prior = float(orders.loc[in_region & (period == prior), "revenue"].sum())
        rev_last = float(orders.loc[in_region & (period == last), "revenue"].sum())

        caveats = []
        if not controls:
            # Every region moved, so there is nothing unchanged to compare with.
            effect = ci_lo = ci_hi = z = raw_seg = raw_ctl = float("nan")
            p_value = 1.0
            strength = "weak"
            caveats.append(
                "no region kept typical spend, so there is no comparison group "
                "for this region"
            )
        else:
            (effect, ci_lo, ci_hi, z, p_value, raw_seg, raw_ctl) = _adjusted_order_effect(
                orders, in_region, in_control, "category", period, last, prior
            )
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

        candidate = moved and strength != "weak"
        if not controls:
            hypothesis = (
                f"In {region}, marketing spend moved {_signed(own_change)}, but no "
                f"region kept typical spend to compare orders against"
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

        results.append(Evidence(
            id=f"stat_marketing_{region}",
            hypothesis=hypothesis,
            evidence_type="statistical",
            metric="orders_change_vs_comparable_segments_pct",
            value=None if not controls else round(effect, 1),
            baseline=0.0,
            segment=f"{region_col}={region}",
            strength=strength,
            sample_size=seg_orders_last,
            caveats=caveats,
            details={
                "period": str(last),
                "prior_period": str(prior),
                "driver": "marketing_spend",
                "driver_moved": bool(moved),
                "driver_change_pct": None if math.isnan(own_change) else round(own_change, 1),
                "driver_typical_change_pct": round(typical, 1),
                "control_segments": controls,
                "top_channel": top_channel,
                "raw_orders_change_pct": None if not controls else round(raw_seg, 1),
                "control_orders_change_pct": None if not controls else round(raw_ctl, 1),
                "ci_low_pct": None if not controls else round(ci_lo, 1),
                "ci_high_pct": None if not controls else round(ci_hi, 1),
                "z": None if not controls else round(z, 2),
                "p_value": None if not controls else float(f"{p_value:.3g}"),
                "p_value_text": None if not controls else _p_text(p_value),
                "spend_prior": round(spend_prior, 2),
                "spend_last": round(spend_last, 2),
                "revenue_prior": round(rev_prior, 2),
                "revenue_last": round(rev_last, 2),
                # Assumes average order value is unchanged, so an order-volume
                # shortfall maps one-for-one onto revenue.
                "revenue_at_stake": _revenue_at_stake(rev_prior, effect) if candidate else None,
            },
        ))

    return _sorted(results)


def _like_for_like_price_change(orders, category_col, period, last, prior):
    """Revenue-weighted change in each product's average unit price between
    the two periods, per category. Only products sold in both periods count,
    so a shift in product mix does not masquerade as a price move."""
    scoped = orders[period.isin([last, prior])].assign(_period=period)
    price = scoped.pivot_table(
        index=["_period", category_col, "product_id"],
        values="unit_price", aggfunc="mean",
    ).reset_index()
    wide = price.pivot_table(
        index=[category_col, "product_id"], columns="_period", values="unit_price"
    ).dropna()
    weights = (
        scoped[scoped["_period"] == prior]
        .groupby(["product_id"])["revenue"].sum()
    )
    wide = wide.join(weights.rename("w"), on="product_id")
    wide["chg"] = (wide[last] / wide[prior] - 1) * 100
    out = {}
    for cat, grp in wide.groupby(level=0):
        out[cat] = float(np.average(grp["chg"], weights=grp["w"])) if grp["w"].sum() else float("nan")
    return out


def price_effect(orders, category_col="category"):
    """
    Test "a price change explains the order-volume change" for each product
    category. One Evidence object per category, strongest first.

    Price change is like-for-like (same products in both periods). A category
    is only a candidate if its price moved away from the typical category
    change; its orders are then compared with categories whose prices did not
    move, adjusted for regional mix.
    """
    period, last, prior = _last_two_periods(orders)
    price_change = _like_for_like_price_change(orders, category_col, period, last, prior)

    results = []
    for category in sorted(orders[category_col].unique()):
        typical, moved, controls = _pick_controls(price_change, category, MATERIAL_PRICE_CHANGE)
        in_cat = orders[category_col] == category
        in_control = orders[category_col].isin(controls)
        seg_orders_last = int((in_cat & (period == last)).sum())

        own_change = price_change.get(category, float("nan"))
        rev_prior = float(orders.loc[in_cat & (period == prior), "revenue"].sum())
        rev_last = float(orders.loc[in_cat & (period == last), "revenue"].sum())

        caveats = []
        net_revenue_effect = float("nan")
        if not controls:
            effect = ci_lo = ci_hi = z = raw_seg = raw_ctl = float("nan")
            p_value = 1.0
            strength = "weak"
            caveats.append(
                "no category kept typical prices, so there is no comparison "
                "group for this category"
            )
        else:
            (effect, ci_lo, ci_hi, z, p_value, raw_seg, raw_ctl) = _adjusted_order_effect(
                orders, in_cat, in_control, "region", period, last, prior
            )
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

        candidate = moved and strength != "weak"
        if not controls:
            hypothesis = (
                f"In {category}, prices moved {_signed(own_change)}, but no category "
                f"kept typical prices to compare orders against"
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

        results.append(Evidence(
            id=f"stat_price_{category}",
            hypothesis=hypothesis,
            evidence_type="statistical",
            metric="orders_change_vs_comparable_segments_pct",
            value=None if not controls else round(effect, 1),
            baseline=0.0,
            segment=f"{category_col}={category}",
            strength=strength,
            sample_size=seg_orders_last,
            caveats=caveats,
            details={
                "period": str(last),
                "prior_period": str(prior),
                "driver": "unit_price",
                "driver_moved": bool(moved),
                "driver_change_pct": None if math.isnan(own_change) else round(own_change, 1),
                "driver_typical_change_pct": round(typical, 1),
                "control_segments": controls,
                "raw_orders_change_pct": None if not controls else round(raw_seg, 1),
                "control_orders_change_pct": None if not controls else round(raw_ctl, 1),
                "ci_low_pct": None if not controls else round(ci_lo, 1),
                "ci_high_pct": None if not controls else round(ci_hi, 1),
                "z": None if not controls else round(z, 2),
                "p_value": None if not controls else float(f"{p_value:.3g}"),
                "p_value_text": None if not controls else _p_text(p_value),
                "revenue_prior": round(rev_prior, 2),
                "revenue_last": round(rev_last, 2),
                "net_revenue_effect_pct": None if math.isnan(net_revenue_effect) else round(net_revenue_effect, 1),
                "revenue_at_stake": _revenue_at_stake(rev_prior, net_revenue_effect) if candidate else None,
            },
        ))

    return _sorted(results)
