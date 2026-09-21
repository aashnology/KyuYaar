"""
Evidence engine core -- Layer 1.

Two entry points:
  baseline_trend()     -- is the metric's latest period down vs. its own
                           history, and by how much?
  segment_breakdown()  -- if it is, which values of a dimension (region,
                           category, ...) are disproportionately responsible?

Both are plain deterministic pandas -- no LLM involved anywhere in this
file. In later layers, the LLM's job is only to decide which of these to
call and how to narrate the result; it never computes a number itself.
"""

from dataclasses import dataclass, field
import pandas as pd

MIN_SAMPLE_SIZE = 30
# Below this total change, "share of the change" is dividing by something close
# to zero and the ranking is noise. Same bar baseline_trend uses for "moderate".
MIN_TOTAL_CHANGE_PCT = 10.0
from strength import STRENGTH_ORDER

_STRENGTH_ORDER = list(STRENGTH_ORDER)


@dataclass
class Evidence:
    id: str
    hypothesis: str
    evidence_type: str          # "observation" | "association" | "statistical" | "decomposition" | "channel"
    metric: str
    value: float | None
    baseline: float | None
    segment: str | None
    strength: str                # "weak" | "moderate" | "strong"
    sample_size: int
    caveats: list[str] = field(default_factory=list)
    # Structured numbers behind the headline value (period labels, control
    # group figures, p-values). Kept separate so downstream code never has
    # to parse the hypothesis string.
    details: dict = field(default_factory=dict)


def _downgrade_if_small_sample(strength: str, sample_size: int) -> tuple[str, list[str]]:
    """Structural 'insufficient data' guardrail: a small sample can't
    support a strong claim, regardless of how large the effect looks."""
    caveats = []
    if sample_size < MIN_SAMPLE_SIZE:
        caveats.append(
            f"sample size ({sample_size}) is below the {MIN_SAMPLE_SIZE}-order "
            f"minimum -- strength downgraded one level"
        )
        idx = max(_STRENGTH_ORDER.index(strength) - 1, 0)
        strength = _STRENGTH_ORDER[idx]
    return strength, caveats


def baseline_trend(df, metric_col="revenue", date_col="order_date", freq="ME"):
    """
    Compare the latest period's total `metric_col` against the baseline
    established by all prior periods.

    Returns a single Evidence object of type "observation" -- this only
    establishes THAT something changed, not WHY. segment_breakdown()
    handles the "why".
    """
    dates = pd.to_datetime(df[date_col])
    ts = df.set_index(dates)[metric_col].resample(freq).sum()
    counts = df.set_index(dates)[metric_col].resample(freq).count()

    if len(ts) < 2:
        raise ValueError("need at least 2 periods to establish a baseline")

    latest_value = ts.iloc[-1]
    history = ts.iloc[:-1]
    baseline_mean = history.mean()
    baseline_std = history.std(ddof=0)

    pct_change = (latest_value - baseline_mean) / baseline_mean * 100

    # Magnitude-based strength: this is "is this worth investigating
    # further", not a significance test -- statistical rigor comes in via
    # segment-level effect sizes and the sample-size guardrail below.
    if abs(pct_change) >= 20:
        strength = "strong"
    elif abs(pct_change) >= 10:
        strength = "moderate"
    else:
        strength = "weak"

    sample_size = int(counts.iloc[-1])
    strength, caveats = _downgrade_if_small_sample(strength, sample_size)

    if baseline_std and abs(latest_value - baseline_mean) < baseline_std:
        caveats.append(
            "latest period is within 1 std dev of the historical baseline "
            "-- could be normal variance, not a real shift"
        )

    direction = "down" if pct_change < 0 else "up"
    return Evidence(
        id="obs_baseline_trend",
        hypothesis=(
            f"Total {metric_col} is {direction} {abs(pct_change):.1f}% vs. "
            f"its {len(history)}-period baseline"
        ),
        evidence_type="observation",
        metric=f"{metric_col}_pct_change_vs_baseline",
        value=round(pct_change, 2),
        baseline=round(baseline_mean, 2),
        segment=None,
        strength=strength,
        sample_size=sample_size,
        caveats=caveats,
        details={
            "latest_period": str(ts.index[-1].to_period(freq[:-1] if freq.endswith("E") else freq)),
            "latest_value": round(float(latest_value), 2),
            "baseline_periods": int(len(history)),
        },
    )


def segment_breakdown(df, dimension_col, metric_col="revenue", date_col="order_date", freq="ME"):
    """
    Break the latest period's `metric_col` down by `dimension_col` (e.g.
    "region" or "category") and rank segments by how disproportionately
    they contributed to the change vs. the prior period.

    "Disproportionate" means: this segment's share of the total change is
    far from its share of the prior baseline. A segment that's simply big
    contributing a big chunk of the drop isn't noteworthy if it was
    already that big a chunk of revenue -- what matters is a segment
    dropping (or rising) far more than its size alone would predict.

    Returns a list of Evidence objects, type "association", sorted with
    the most disproportionate segment first.
    """
    work = df.copy()
    period_freq = freq[:-1] if freq.endswith("E") else freq  # to_period wants 'M', resample wants 'ME'
    work["_period"] = pd.to_datetime(work[date_col]).dt.to_period(period_freq)
    periods = sorted(work["_period"].unique())
    if len(periods) < 2:
        raise ValueError("need at least 2 periods to compare segments")

    last_period, prior_period = periods[-1], periods[-2]

    pivot = work.pivot_table(
        index="_period", columns=dimension_col, values=metric_col,
        aggfunc="sum", fill_value=0,
    )
    counts = work.pivot_table(
        index="_period", columns=dimension_col, values=metric_col,
        aggfunc="count", fill_value=0,
    )

    last_vals, prior_vals = pivot.loc[last_period], pivot.loc[prior_period]
    last_counts = counts.loc[last_period]
    total_last, total_prior = last_vals.sum(), prior_vals.sum()
    total_delta = total_last - total_prior
    total_change_pct = (total_delta / total_prior * 100) if total_prior else 0.0
    shares_unstable = abs(total_change_pct) < MIN_TOTAL_CHANGE_PCT

    ranked = []
    for segment in pivot.columns:
        seg_delta = last_vals[segment] - prior_vals[segment]
        seg_pct_change = (
            seg_delta / prior_vals[segment] * 100 if prior_vals[segment] else float("nan")
        )

        share_of_total_delta = (seg_delta / total_delta) if total_delta else 0.0
        share_of_baseline = (prior_vals[segment] / total_prior) if total_prior else 0.0
        disproportionality = share_of_total_delta - share_of_baseline

        # Relative excess, not a flat point-gap: a segment that's already
        # a big chunk of baseline revenue needs a much bigger point-gap to
        # clear the same bar as a small segment, which understates exactly
        # the segments most worth flagging. A segment taking LESS than its
        # fair share of the change isn't a cause (it may even be a bright
        # spot), so only positive excess counts toward strength.
        if share_of_baseline > 0:
            relative_excess = disproportionality / share_of_baseline
        else:
            relative_excess = float("inf") if disproportionality > 0 else 0.0

        if relative_excess <= 0:
            strength = "weak"
        elif relative_excess >= 0.50:
            strength = "strong"
        elif relative_excess >= 0.20:
            strength = "moderate"
        else:
            strength = "weak"

        sample_size = int(last_counts[segment])
        strength, caveats = _downgrade_if_small_sample(strength, sample_size)
        if shares_unstable:
            strength = "weak"
            caveats.append(
                f"the overall {metric_col} change is only {total_change_pct:+.1f}%, "
                f"so each segment's share of the change is unstable"
            )

        hyp = (
            f"{dimension_col}={segment} accounts for {share_of_total_delta*100:.0f}% "
            f"of the total {metric_col} change, vs. a {share_of_baseline*100:.0f}% "
            f"share of prior-period {metric_col} "
            f"(its own {metric_col} moved {seg_pct_change:+.1f}%)"
        )

        evidence = Evidence(
            id=f"assoc_{dimension_col}_{segment}",
            hypothesis=hyp,
            evidence_type="association",
            metric=f"{metric_col}_share_of_overall_change",
            value=round(share_of_total_delta * 100, 1),
            baseline=round(share_of_baseline * 100, 1),
            segment=f"{dimension_col}={segment}",
            strength=strength,
            sample_size=sample_size,
            caveats=caveats,
            details={
                "period": str(last_period),
                "prior_period": str(prior_period),
                "segment_pct_change": None if pd.isna(seg_pct_change) else round(float(seg_pct_change), 1),
                "last_value": round(float(last_vals[segment]), 2),
                "prior_value": round(float(prior_vals[segment]), 2),
            },
        )
        ranked.append((abs(disproportionality), evidence))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [evidence for _, evidence in ranked]
