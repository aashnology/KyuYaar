"""
Recommendation -- Layer 8.

recommend() annotates the decision options with one "Recommended" pick, or says
plainly that the evidence does not yet favor one. It computes nothing new: it
reads the evidence strengths, the run_scenario() projections and the risks each
option already states, and applies four rules in order.

1. Evidence gate. An option is only considered if every hypothesis it rests on
   is strong or moderate (read through src/strength.py). Weak evidence never
   supports a recommendation, whatever the projection says. If that rules out
   the option with the largest projected number, the result says so.
2. Projected impact. The measure is the change in gross profit over the
   horizon, which already nets off the marketing spend an option adds or the
   margin a price rollback gives up. Revenue recovered is shown next to it but
   is not the ranking measure: restoring spend can win back revenue and still
   lose money, and recommending it on the revenue figure would hide that.
   Options that project no gain under the person's assumptions are not
   recommended. The hold baseline and options run_scenario() cannot size
   (sequence_fixes) are not ranked either.
3. Risk adjustment. Each option gets a risk tier from three checkable things
   (below), and its projected impact is multiplied by that tier's factor before
   ranking. A high-risk option therefore has to project twice what a low-risk
   one does to outrank it. Ties break on stronger evidence, then option id, so
   the result is the same on every run.
4. One option is marked recommended, with a sentence tying it to its evidence
   strength and its projected impact.

Risk points (capped at two):
    +1  the option is a full commitment ("act"), not a bounded trial ("test")
    +1  the 95% interval on its projected gross profit reaches zero or below,
        or no interval is available, so a loss is within what the data allows
    +1  for each risk the option itself states beyond the usual two

The tiers and factors are judgment calls, not estimates from the data. They are
constants here, in one place, so they can be argued with and changed.
"""

from dataclasses import dataclass, field

from strength import BRIEF_NAME, RANK, WORD, is_actionable, weakest

# Points -> tier -> factor applied to projected impact before ranking.
RISK_TIERS = ("low", "medium", "high")
RISK_FACTOR = {"low": 1.0, "medium": 0.75, "high": 0.5}
USUAL_STATED_RISKS = 2

CLOSING_LINE = (
    "This is a recommendation based on the available evidence, not a decision. "
    "The final choice is yours."
)

# Overall result.
RECOMMENDED = "recommended"
NO_EVIDENCE = "insufficient_evidence"
NO_GAIN = "no_projected_gain"

# What happened to each option.
BASELINE = "baseline"       # the hold option: what the others are measured against
EVIDENCE = "evidence"       # failed the evidence gate
UNSIZED = "unsized"         # no gross profit projection to rank on
NO_PROJECTED_GAIN = "no_gain"
RANKED = "ranked"


@dataclass
class Assessment:
    option_id: str
    title: str
    evidence_strength: str              # weakest strength behind it, or "n/a"
    outcome: str                        # BASELINE | EVIDENCE | UNSIZED | NO_GAIN | RANKED
    reason: str                         # why it was left out, or how it was ranked
    impact: float | None = None         # gross profit change over the horizon
    revenue: float | None = None
    risk_tier: str | None = None
    risk_factor: float | None = None
    risk_reasons: list[str] = field(default_factory=list)
    adjusted_impact: float | None = None
    rank: int | None = None


@dataclass
class Recommendation:
    status: str
    recommended_id: str | None
    sentence: str
    ranking: list[str]                  # option ids, best first; empty when unranked
    assessments: dict[str, Assessment]
    notes: list[str] = field(default_factory=list)
    closing: str = CLOSING_LINE

    @property
    def has_pick(self) -> bool:
        return self.recommended_id is not None


# ------------------------------------------------------------- helpers ---

def _money(x):
    return f"{x:,.0f}"


def project_all(decision_set, evidence, orders, assumptions_for=None):
    """run_scenario() for every option. `assumptions_for(option)` supplies the
    person's assumptions for that option; None uses the placeholders."""
    from scenario import run_scenario
    return {
        o.id: run_scenario(o, evidence, orders, assumptions_for(o) if assumptions_for else None)
        for o in decision_set.options
    }


def _evidence_strength(option, by_id):
    """(weakest strength behind the option, problem or None). Read from the
    evidence itself, not from the option's own confidence string."""
    if not option.addresses:
        return "n/a", "it does not rest on any tested hypothesis"
    missing = [i for i in option.addresses if i not in by_id]
    if missing:
        return "n/a", f"its evidence ({', '.join(missing)}) is not in this investigation"
    return weakest(by_id[i].strength for i in option.addresses), None


def risk_of(option, scenario):
    points, reasons = 0, []
    if option.kind == "act":
        points += 1
        reasons.append("a full commitment rather than a bounded test")
    rng = scenario.gross_profit_range
    if rng is None:
        points += 1
        reasons.append("no interval on its projected gross profit")
    elif rng[0] <= 0:
        points += 1
        reasons.append(
            f"its 95% interval on gross profit runs from {rng[0]:+,.0f} to {rng[1]:+,.0f}, "
            "so a loss is within what the data allows"
        )
    extra = len(option.risks) - USUAL_STATED_RISKS
    if extra > 0:
        points += extra
        reasons.append(f"it states {len(option.risks)} risks, {extra} more than usual")
    tier = RISK_TIERS[min(points, len(RISK_TIERS) - 1)]
    return tier, RISK_FACTOR[tier], reasons


def _assess(option, scenario, by_id):
    strength, problem = _evidence_strength(option, by_id)
    a = Assessment(option.id, option.title, strength, outcome=EVIDENCE, reason="")
    if scenario is not None and scenario.projectable and scenario.gross_profit_total is not None:
        a.impact, a.revenue = scenario.gross_profit_total, scenario.revenue_total

    if option.kind == "hold":
        a.outcome = BASELINE
        a.reason = "The no-action baseline; it is what the other options are measured against."
    elif problem:
        a.reason = f"Not considered: {problem}."
    elif not is_actionable(strength):
        a.reason = (f"Not considered: the evidence behind it is {WORD[strength].lower()} "
                    f"({BRIEF_NAME[strength]}). Weak evidence never supports a recommendation.")
    elif scenario is None or not scenario.projectable:
        a.outcome = UNSIZED
        a.reason = "Not ranked: it cannot be sized without double counting, so it has no projected impact."
    elif scenario.gross_profit_total is None:
        a.outcome = UNSIZED
        a.reason = "Not ranked: no gross-profit projection is available for it."
    elif a.impact <= 0:
        a.outcome = NO_PROJECTED_GAIN
        a.reason = (f"Not recommended: it projects a gross profit change of {a.impact:+,.0f} "
                    f"under the current assumptions, which is not a gain.")
    else:
        a.outcome = RANKED
        a.risk_tier, a.risk_factor, a.risk_reasons = risk_of(option, scenario)
        a.adjusted_impact = a.impact * a.risk_factor
    return a


# --------------------------------------------------------- entry point ---

def recommend(decision_set, evidence, scenarios) -> Recommendation:
    """Annotate the options with a recommendation. `scenarios` maps option id to
    its run_scenario() result. The options themselves are not changed."""
    by_id = {e.id: e for e in evidence}
    assessed = {o.id: _assess(o, scenarios.get(o.id), by_id) for o in decision_set.options}
    notes = []

    # The largest projected number among the options that carry one. If the
    # evidence gate is what removed it, say so.
    sized = [a for a in assessed.values() if a.impact is not None and a.outcome != BASELINE]
    top_raw = max(sized, key=lambda a: a.impact, default=None)
    if top_raw is not None and top_raw.impact > 0 and top_raw.outcome == EVIDENCE:
        notes.append(
            f"The largest projected impact belongs to \"{top_raw.title}\" "
            f"({top_raw.impact:+,.0f} gross profit), but it is excluded. {top_raw.reason}"
        )

    eligible = sorted(
        (a for a in assessed.values() if a.outcome == RANKED),
        key=lambda a: (-a.adjusted_impact, RANK[a.evidence_strength], a.option_id),
    )

    if not eligible:
        passed_gate = [a for a in assessed.values() if a.outcome in (UNSIZED, NO_PROJECTED_GAIN)]
        if passed_gate:
            sentence = (
                "The evidence supports at least one option, but none projects a gross profit gain "
                "under your current assumptions, so none is recommended and the options are left unranked."
            )
            notes.append(
                "A longer horizon, a larger share won back or a shorter lag can change this; "
                "each option's break-even share shows how far it is from paying for itself."
            )
            return Recommendation(NO_GAIN, None, sentence, [], assessed, notes)
        sentence = (
            "The evidence does not yet favor one option over another, so no option is "
            "recommended and the options are left unranked."
        )
        return Recommendation(NO_EVIDENCE, None, sentence, [], assessed, notes)

    for i, a in enumerate(eligible, start=1):
        a.rank = i
        a.reason = (
            f"Ranked {i} of {len(eligible)}: {a.impact:+,.0f} projected, "
            f"{a.adjusted_impact:+,.0f} after the {a.risk_tier}-risk adjustment (x{a.risk_factor:g})."
        )

    best = eligible[0]
    raw_leader = max(eligible, key=lambda a: a.impact)
    if raw_leader.option_id != best.option_id:
        notes.append(
            f"\"{raw_leader.title}\" projects more ({raw_leader.impact:+,.0f}), but its {raw_leader.risk_tier} "
            f"risk discounts it to {raw_leader.adjusted_impact:+,.0f}, below \"{best.title}\" "
            f"at {best.adjusted_impact:+,.0f}."
        )

    horizon = scenarios[best.option_id].assumptions.horizon_months
    sentence = (
        f"Recommended: {best.title}. The evidence behind it is {WORD[best.evidence_strength].lower()} "
        f"({BRIEF_NAME[best.evidence_strength]}), and under the current assumptions it projects a gross profit change of "
        f"{best.impact:+,.0f} over {horizon} month{'s' if horizon != 1 else ''} "
        f"({best.adjusted_impact:+,.0f} after adjusting for {best.risk_tier} risk)."
    )
    return Recommendation(RECOMMENDED, best.option_id, sentence, [a.option_id for a in eligible], assessed, notes)

