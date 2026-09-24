"""
Marketing by channel -- Layer 4.

marketing_effect() tests whether a region's total marketing spend moved with
its orders. This module asks the same question one level down, for each paid
channel in each region (Paid in North, Email in West, ...), and adds two checks
the regional test cannot make.

  Is the loss concentrated in the channel whose spend moved?
      A cut in Paid spend should cost Paid orders. If orders in the other
      channels (Organic, which has no spend, Email, Referral) fell by as much,
      the decline is region-wide and the channel data cannot show that Paid
      spend caused it.

  Did the channel get less effective, or was less bought?
      Cost per attributed order, before and after. When spend and orders fall
      together the cost per order stays put: the channel performed as before,
      there was just less of it.

Method
------
A cell (one channel in one region) is a candidate only if its spend moved away
from the typical change for that channel across regions. Its orders, meaning
orders recorded under that channel, are then compared with the same channel in
comparison regions, inside each product category, using the pooled log rate
ratio from effects.py. Comparison regions are those where spend stayed typical
in every paid channel, so the same set can also serve the specificity test.

Specificity compares two effects measured against the same comparison regions:
the cell's own orders, and the orders in all other channels of that region.
The difference of the two log effects is tested with the usual normal
approximation. The two sets of orders do not overlap, so the effects are
independent.

A cell whose test cannot be run (no spend in the channel in that region in
either month, no spend to measure a change from, no comparison regions, no
orders on one side) gets an insufficient-data Evidence instead of an error, the
same way effects.py does. The specificity check is optional detail: when it
cannot be computed it is left out rather than failing the cell.

Nothing here establishes cause. A spend change and an order change in the same
month leave room for something else that changed at the same time, and
channel attribution adds its own uncertainty; both are stated on the evidence.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from effects import (
    MATERIAL_SPEND_CHANGE, _TIMING_CAVEAT, InsufficientData, _adjusted_order_effect,
    _grade, _insufficient_caveats, _insufficient_details, _last_two_periods, _p_text,
    _pct, _pick_controls, _signed, _sorted, _undefined_change_reason,
)
from evidence import Evidence, _downgrade_if_small_sample

SPECIFICITY_P = 0.05        # p-value bar for "the cell moved more than its region's other channels"
EFFICIENCY_MOVE_PCT = 10.0  # cost per order must move at least this much, and clear EFFICIENCY_P
EFFICIENCY_P = 0.05

_ATTRIBUTION_CAVEAT = (
    "orders are assigned to channels by the recorded channel field, so a channel's "
    "orders can move for reasons unrelated to its own spend, for example a customer "
    "reached through one channel but recorded under another"
)


def _spend_table(marketing, last, prior):
    """{(channel, region): (spend_prior, spend_last)} for the two periods."""
    period = pd.to_datetime(marketing["date"]).dt.to_period("M")
    scoped = marketing.assign(_period=period)[period.isin([last, prior])]
    table = (
        scoped.pivot_table(index=["channel", "region"], columns="_period", values="spend", aggfunc="sum")
        .reindex(columns=[prior, last])
        .fillna(0.0)
    )
    return {key: (float(row[prior]), float(row[last])) for key, row in table.iterrows()}


def _quiet_regions(spend_change, regions, target):
    """Regions other than `target` whose spend stayed typical in every paid channel."""
    typical = {}
    for channel, changes in spend_change.items():
        valid = [v for v in changes.values() if not math.isnan(v)]
        typical[channel] = float(np.median(valid)) if valid else 0.0
    quiet = []
    for region in regions:
        if region == target:
            continue
        moves = {c: spend_change[c].get(region, float("nan")) for c in spend_change}
        if any(math.isnan(m) for m in moves.values()):
            continue
        if all(abs(m - typical[c]) < MATERIAL_SPEND_CHANGE for c, m in moves.items()):
            quiet.append(region)
    return quiet


def _log_effect(effect_pct, ci_low_pct, ci_high_pct):
    """Pooled log rate ratio and its standard error, recovered from the
    percent-scale effect and 95% interval that effects.py returns."""
    if min(effect_pct, ci_low_pct, ci_high_pct) <= -100 or not math.isfinite(ci_high_pct):
        raise InsufficientData("degenerate_estimate", "the estimated order change is not a usable percentage")
    pooled = math.log1p(effect_pct / 100)
    se = (math.log1p(ci_high_pct / 100) - math.log1p(ci_low_pct / 100)) / (2 * 1.96)
    return pooled, se


def _two_sided(z):
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def _efficiency(channel, spend_prior, spend_last, n_prior, n_last):
    """Cost per attributed order before and after. Spend is taken as fixed and
    only the order count carries sampling noise, treated as Poisson."""
    if min(n_prior, n_last) == 0 or spend_prior <= 0 or spend_last <= 0:
        return None
    cpo0, cpo1 = spend_prior / n_prior, spend_last / n_last
    change = (cpo1 / cpo0 - 1) * 100
    z = math.log(cpo1 / cpo0) / math.sqrt(1 / n_prior + 1 / n_last)
    p = _two_sided(z)
    moved = abs(change) >= EFFICIENCY_MOVE_PCT and p < EFFICIENCY_P
    verdict = "unchanged" if not moved else ("worse" if change > 0 else "better")

    cpo0, cpo1, change_r, p_text = round(cpo0, 2), round(cpo1, 2), round(change, 1), _p_text(p)
    lead = (
        f"Cost per {channel} order went from {cpo0:,.2f} to {cpo1:,.2f} "
        f"({change_r:+.1f}%, p {p_text})"
    )
    if verdict == "unchanged":
        text = (f"{lead}, which is not distinguishable from ordinary variation, so {channel} "
                f"orders moved roughly in proportion to its spend.")
    elif verdict == "worse":
        text = (f"{lead}: each unit of {channel} spend bought fewer orders than before, so the "
                f"channel also became less effective, not only smaller.")
    else:
        text = (f"{lead}: each unit of {channel} spend bought more orders than before, so "
                f"orders moved by less than spend did.")
    return {
        "cost_per_order_prior": cpo0, "cost_per_order_last": cpo1,
        "cost_per_order_change_pct": change_r,
        "p_value": float(f"{p:.3g}"), "p_value_text": p_text,
        "verdict": verdict, "text": text,
    }


def _specificity(orders, region, channel, controls, cell_fit, gap, period, last, prior,
                 region_col, channel_col, cell_effect_pct):
    """Compare the cell's order effect with the effect on the region's other
    channels, both measured against the same comparison regions."""
    in_other = (orders[region_col] == region) & (orders[channel_col] != channel)
    in_other_ctl = orders[region_col].isin(controls) & (orders[channel_col] != channel)
    others = sorted(orders.loc[in_other, channel_col].dropna().unique())
    if not others or not in_other.any():
        return None

    try:
        eo, lo, hi, _, p_o, _, _ = _adjusted_order_effect(
            orders, in_other, in_other_ctl, "category", period, last, prior
        )
        pooled_c, se_c = cell_fit
        pooled_o, se_o = _log_effect(eo, lo, hi)
    except InsufficientData:
        return None
    combined = math.sqrt(se_c ** 2 + se_o ** 2)
    if not math.isfinite(combined) or combined <= 0:
        return None
    diff = pooled_c - pooled_o
    z = diff / combined
    p_gap = _two_sided(z)

    expected = -1 if gap < 0 else 1            # a spend cut should lower orders, an increase raise them
    others_grade = _grade(True, True, eo, p_o) if eo * expected > 0 else "weak"
    if diff * expected > 0 and p_gap < SPECIFICITY_P:
        status = "concentrated"
    elif others_grade != "weak":
        status = "region_wide"
    else:
        status = "unclear"

    cell_r, other_r, p_text = round(cell_effect_pct, 1), round(eo, 1), _p_text(p_gap)
    names = ", ".join(others)
    if status == "concentrated":
        text = (f"In {region}, {channel} orders changed {cell_r:+.1f}% while orders in the other channels "
                f"({names}) changed {other_r:+.1f}% (difference p {p_text}): the loss is concentrated "
                f"in the channel whose spend moved, as a channel-specific effect predicts.")
    elif status == "region_wide":
        text = (f"In {region}, orders in the other channels ({names}) changed {other_r:+.1f}% against "
                f"regions with typical spend, close to {channel} ({cell_r:+.1f}%; difference p {p_text}). "
                f"The change is region-wide rather than concentrated in the channel whose spend moved, "
                f"so this data cannot show whether {channel} spend drove it.")
    else:
        text = (f"In {region}, {channel} orders changed {cell_r:+.1f}% and orders in the other channels "
                f"({names}) changed {other_r:+.1f}% (difference p {p_text}). The data cannot tell "
                f"whether the change is specific to {channel}.")
    return {
        "status": status, "text": text,
        "other_channels": others,
        "other_orders_change_pct": other_r,
        "other_ci_low_pct": round(lo, 1), "other_ci_high_pct": round(hi, 1),
        "other_p_value_text": _p_text(p_o),
        "difference_p_value": float(f"{p_gap:.3g}"), "difference_p_value_text": p_text,
    }


def marketing_channel_analysis(orders, marketing, region_col="region", channel_col="channel"):
    """
    Test "a change in this channel's spend explains the order change" for each
    paid channel in each region. One Evidence object (type "channel") per
    region and channel, strongest first.

    Strength follows the Layer 2 rule for the cell's own orders. For a cell
    whose spend moved, details also carry the specificity check (is the loss
    concentrated in this channel, or region-wide?) and the efficiency check
    (cost per attributed order), each with a ready-made sentence.

    A cell that cannot be tested returns an insufficient-data Evidence (see
    effects.py) rather than raising.
    """
    if channel_col not in orders.columns:
        raise ValueError(f"orders needs a '{channel_col}' column for channel analysis")

    period, last, prior = _last_two_periods(orders)
    regions = sorted(orders[region_col].dropna().unique())
    spend = _spend_table(marketing, last, prior)
    paid = sorted({c for (c, _), (a, b) in spend.items() if a + b > 0})
    if not paid:
        return []

    spend_change = {
        c: {r: _pct(*reversed(spend.get((c, r), (0.0, 0.0)))) for r in regions}
        for c in paid
    }

    results = []
    for channel in paid:
        for region in regions:
            typical, moved, _ = _pick_controls(spend_change[channel], region, MATERIAL_SPEND_CHANGE)
            controls = _quiet_regions(spend_change, regions, region)
            own_change = spend_change[channel][region]
            spend_prior, spend_last = spend.get((channel, region), (0.0, 0.0))

            in_cell = (orders[region_col] == region) & (orders[channel_col] == channel)
            in_ctl = orders[region_col].isin(controls) & (orders[channel_col] == channel)
            n_prior = int((in_cell & (period == prior)).sum())
            n_last = int((in_cell & (period == last)).sum())

            insufficient = _undefined_change_reason(
                f"{channel} spend in {region}", spend_prior, spend_last, str(prior), str(last)
            )
            if insufficient is None and not controls:
                insufficient = (
                    "no_comparison_group",
                    "no region kept typical spend in every paid channel, so there is no "
                    "comparison group for this channel and region",
                )
            if insufficient is None:
                try:
                    effect, ci_lo, ci_hi, z, p_value, raw_seg, raw_ctl = _adjusted_order_effect(
                        orders, in_cell, in_ctl, "category", period, last, prior
                    )
                except InsufficientData as exc:
                    insufficient = (exc.code, exc.reason)

            caveats, pattern, efficiency = [], None, None
            if insufficient is not None:
                effect = ci_lo = ci_hi = z = raw_seg = raw_ctl = float("nan")
                p_value, strength = 1.0, "weak"
                caveats.extend(_insufficient_caveats(*insufficient))
            else:
                gap = own_change - typical
                sign_ok = moved and ((gap < 0 and effect < 0) or (gap > 0 and effect > 0))
                strength = _grade(moved, sign_ok, effect, p_value)
                strength, small = _downgrade_if_small_sample(strength, n_last)
                caveats.extend(small)

                if not moved:
                    caveats.append(
                        f"{channel} spend in this region stayed close to the typical change, so "
                        f"it is not a candidate explanation here"
                    )
                elif not sign_ok:
                    caveats.append(
                        "orders moved in the opposite direction to what the spend change would predict"
                    )
                else:
                    caveats += [_TIMING_CAVEAT, _ATTRIBUTION_CAVEAT]

                if moved:
                    try:
                        pattern = _specificity(
                            orders, region, channel, controls, _log_effect(effect, ci_lo, ci_hi), gap,
                            period, last, prior, region_col, channel_col, effect,
                        )
                    except InsufficientData:
                        pattern = None
                    efficiency = _efficiency(channel, spend_prior, spend_last, n_prior, n_last)

            if insufficient is not None and insufficient[0] == "no_comparison_group":
                hypothesis = (
                    f"In {region}, {channel} spend moved {_signed(own_change)}, but no region kept "
                    f"typical spend to compare {channel} orders against"
                )
            elif insufficient is not None:
                hypothesis = (
                    f"In {region}, {channel} spend cannot be tested as an explanation "
                    f"(insufficient data: {insufficient[1]})"
                )
            elif moved:
                kind = "cut" if own_change < typical else "increase"
                hypothesis = (
                    f"In {region}, a {kind} in {channel} spend ({_signed(own_change)} vs. "
                    f"{_signed(typical)} typical elsewhere) accompanies a {_signed(effect)} change "
                    f"in {channel}-attributed orders relative to regions with typical {channel} spend"
                )
            else:
                hypothesis = (
                    f"In {region}, {channel} spend stayed in line with other regions "
                    f"({_signed(own_change)} vs. {_signed(typical)}); {channel}-attributed orders "
                    f"changed {_signed(effect)} relative to regions with typical {channel} spend"
                )

            has_fit = insufficient is None
            details = {
                "period": str(last), "prior_period": str(prior),
                "driver": "channel_spend",
                "region": region, "channel": channel,
                "driver_moved": bool(moved),
                "driver_change_pct": None if math.isnan(own_change) else round(own_change, 1),
                "driver_typical_change_pct": round(typical, 1),
                "control_segments": controls,
                "spend_prior": round(spend_prior, 2), "spend_last": round(spend_last, 2),
                "orders_prior": n_prior, "orders_last": n_last,
                "raw_orders_change_pct": round(raw_seg, 1) if has_fit else None,
                "control_orders_change_pct": round(raw_ctl, 1) if has_fit else None,
                "ci_low_pct": round(ci_lo, 1) if has_fit else None,
                "ci_high_pct": round(ci_hi, 1) if has_fit else None,
                "z": round(z, 2) if has_fit else None,
                "p_value": float(f"{p_value:.3g}") if has_fit else None,
                "p_value_text": _p_text(p_value) if has_fit else None,
                "channel_pattern": pattern,
                "efficiency": efficiency,
            }
            if insufficient is not None:
                details.update(_insufficient_details(*insufficient))

            results.append(Evidence(
                id=f"chan_{region}_{channel}",
                hypothesis=hypothesis,
                evidence_type="channel",
                metric="channel_orders_change_vs_comparable_regions_pct",
                value=round(effect, 1) if has_fit else None,
                baseline=0.0,
                segment=f"{region_col}={region}|{channel_col}={channel}",
                strength=strength,
                sample_size=n_last,
                caveats=caveats,
                details=details,
            ))

    return _sorted(results)


def channel_check(cause, evidence):
    """
    The channel-level findings behind a supported regional marketing cause.

    `cause` is a marketing_effect Evidence object; `evidence` is everything
    gathered so far. Returns None unless the channel analysis was run and a
    channel in that region moved its spend. Otherwise a dict with the channel,
    the specificity status ("concentrated" | "region_wide" | "unclear"), the
    sentence for it, and the efficiency finding if there is one.
    """
    if not cause.segment or not cause.segment.startswith("region="):
        return None
    region = cause.segment.split("=", 1)[1]
    cells = [
        e for e in evidence
        if e.evidence_type == "channel" and e.details.get("region") == region
        and e.details.get("driver_moved") and e.details.get("channel_pattern")
    ]
    if not cells:
        return None
    cell = _sorted(cells)[0]
    pattern = cell.details["channel_pattern"]
    return {
        "channel": cell.details["channel"],
        "status": pattern["status"],
        "text": pattern["text"],
        "efficiency": cell.details.get("efficiency"),
        "cell": cell,
    }
