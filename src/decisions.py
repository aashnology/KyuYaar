"""
Decision options -- Layer 2.

Options come from templates keyed on which candidate causes the evidence
supports at strong or moderate strength. Everything numeric in an option
(revenue at stake, spend that would be restored) is computed from the
Evidence details, never written by the model.

KyuYaar does not rank options or pick one. Each option states what it
assumes and what it risks; the person choosing owns the call.
"""

from dataclasses import dataclass, field

from decomposition import signature_check
from narration import label

_ACTIONABLE = ("strong", "moderate")


@dataclass
class DecisionOption:
    id: str
    kind: str                     # "act" | "test" | "hold"
    title: str
    rationale: str
    addresses: list[str]
    confidence: str               # weakest strength among the evidence it rests on, or "n/a"
    impact: str
    assumptions: list[str]
    risks: list[str]
    impact_value: float | None = None


@dataclass
class DecisionSet:
    options: list[DecisionOption]
    insufficient: bool
    unresolved: list[str] = field(default_factory=list)
    not_supported: list[str] = field(default_factory=list)


def _money(x):
    return f"{x:,.0f}"


def _name(ev):
    return ev.segment.split("=", 1)[1] if ev.segment and "=" in ev.segment else ev.segment


def _weakest(evs):
    order = {"weak": 0, "moderate": 1, "strong": 2}
    return min((e.strength for e in evs), key=lambda s: order[s]) if evs else "n/a"


def _marketing_options(ev, evidence):
    d, region = ev.details, _name(ev)
    sig = signature_check(ev, evidence)
    cut = max(d["spend_prior"] - d["spend_last"], 0.0)
    stake = d.get("revenue_at_stake") or 0.0
    channel = f" (mainly {d['top_channel']})" if d.get("top_channel") else ""
    impact = (
        f"Up to {_money(stake)} of monthly revenue is at stake in {region} (estimate; "
        f"assumes average order value is unchanged). Restoring spend costs about "
        f"{_money(cut)} a month."
    )
    common_assumptions = [
        "The order decline in this region is driven by the spend cut, not by something else that changed in the same month.",
        "Orders respond to restored spend about as they responded to the cut.",
        "Average order value stays roughly where it is.",
    ]
    if sig:
        # Turn the stated assumption into a checked one where the data allows.
        common_assumptions[2] = (
            f"Average order value stays roughly where it is. Checked against the data: {sig['text']}"
        )
    act = DecisionOption(
        id=f"restore_marketing_{region}", kind="act",
        title=f"Restore marketing spend in {region}",
        rationale=(
            f"Spend in {region} fell{channel} while regions with typical spend held up, "
            f"and {region}'s order volume moved with it."
        ),
        addresses=[ev.id], confidence=ev.strength, impact=impact,
        assumptions=common_assumptions,
        risks=[
            f"Adds about {_money(cut)} a month in cost; if the cut was not the real cause, that spend does not bring orders back.",
            "Recovery may lag the spend by a few weeks, so an early read can look like failure.",
        ],
        impact_value=stake,
    )
    test = DecisionOption(
        id=f"test_marketing_{region}", kind="test",
        title=f"Restore spend in part of {region} first, and compare",
        rationale=(
            "Restores spend for one channel or one sub-area only and leaves the rest as it is, "
            "so the difference in orders shows how much of the drop the spend cut actually explains."
        ),
        addresses=[ev.id], confidence=ev.strength,
        impact=(
            f"Recovers only part of the {_money(stake)} at stake while the test runs, "
            f"but replaces an assumption with a measured effect."
        ),
        assumptions=[
            "Part of the region can be treated differently from the rest without leaking between them.",
            "One period is long enough to see a difference in orders.",
        ],
        risks=["Slower recovery than a full restore.", "A small test group gives a noisier answer."],
        impact_value=0.0,
    )
    return [act, test]


def _price_options(ev, evidence):
    d, cat = ev.details, _name(ev)
    sig = signature_check(ev, evidence)
    pattern = f" {sig['text']}" if sig else ""
    stake = d.get("revenue_at_stake") or 0.0
    increased = (d.get("driver_change_pct") or 0) > (d.get("driver_typical_change_pct") or 0)
    impact = (
        f"Up to {_money(stake)} of monthly revenue is at stake in {cat} (estimate: higher "
        f"price per order times fewer orders, against comparable categories)."
    )
    if increased:
        act = DecisionOption(
            id=f"rollback_price_{cat}", kind="act",
            title=f"Roll back or soften the {cat} price increase",
            rationale=(
                f"{cat} prices rose while comparable categories held steady, and {cat} order "
                f"volume fell against them.{pattern}"
            ),
            addresses=[ev.id], confidence=ev.strength, impact=impact,
            assumptions=[
                "The price increase, not another change, is the main reason orders fell.",
                "Demand returns roughly as far as prices come back down.",
            ],
            risks=[
                "Gives up the extra margin per unit the higher price was earning.",
                "Demand may not fully return, and reversing a price can affect how customers see the brand.",
            ],
            impact_value=stake,
        )
        test = DecisionOption(
            id=f"test_price_{cat}", kind="test",
            title=f"Roll back the price on a subset of {cat} products first",
            rationale=(
                "Reverts prices on a few products and keeps the rest, so the order response "
                "to price is measured directly instead of inferred from one month."
            ),
            addresses=[ev.id], confidence=ev.strength,
            impact="Recovers only part of the revenue at stake during the test, in exchange for a measured price response.",
            assumptions=[
                "The chosen products are typical of the category.",
                "One period is long enough to read the difference.",
            ],
            risks=["Slower recovery than a full rollback.", "Customers may notice mixed pricing inside one category."],
            impact_value=0.0,
        )
        return [act, test]

    review = DecisionOption(
        id=f"review_price_{cat}", kind="act",
        title=f"Review the {cat} price change",
        rationale=f"{cat} prices moved against comparable categories and order volume moved with them.{pattern}",
        addresses=[ev.id], confidence=ev.strength, impact=impact,
        assumptions=["The price change is the main reason order volume moved."],
        risks=["Margin per unit may be lower than before the change."],
        impact_value=stake,
    )
    return [review]


def build_options(evidence) -> DecisionSet:
    causes = [
        e for e in evidence
        if e.evidence_type == "statistical" and e.strength in _ACTIONABLE
    ]
    causes.sort(key=lambda e: -(e.details.get("revenue_at_stake") or 0))

    not_supported = []
    weak_marketing = [_name(e) for e in evidence
                      if e.id.startswith("stat_marketing_") and e.strength == "weak"]
    weak_price = [_name(e) for e in evidence
                  if e.id.startswith("stat_price_") and e.strength == "weak"]
    if weak_marketing:
        not_supported.append("Marketing spend: " + ", ".join(sorted(weak_marketing)))
    if weak_price:
        not_supported.append("Pricing: " + ", ".join(sorted(weak_price)))

    options = []
    for ev in causes:
        if ev.id.startswith("stat_marketing_"):
            options.extend(_marketing_options(ev, evidence))
        elif ev.id.startswith("stat_price_"):
            options.extend(_price_options(ev, evidence))

    unresolved = ["Whether something else that changed in the same month contributed to the drop."]
    stakes = [(_name(e), e.details.get("revenue_at_stake") or 0) for e in causes]

    if len(causes) >= 2:
        segs = " and ".join(_name(e) for e in causes)
        options.append(DecisionOption(
            id="sequence_fixes", kind="test",
            title="Change one lever first, then the other",
            rationale=(
                f"The supported explanations ({segs}) can overlap wherever the same "
                f"customers buy the same products. Changing both at once would make it "
                f"impossible to tell which one brought orders back."
            ),
            addresses=[e.id for e in causes], confidence=_weakest(causes),
            impact="Recovery is slower than acting on both, but each lever's effect becomes measurable.",
            assumptions=["The two effects are roughly additive and can be separated over successive periods."],
            risks=["Revenue stays depressed longer while the second lever waits."],
            impact_value=0.0,
        ))
        unresolved.append(
            "Whether the explanations interact where they overlap; this data cannot separate "
            "their combined effect from their individual effects."
        )

    if stakes:
        listed = ", ".join(f"{_money(v)} in {n}" for n, v in stakes)
        overlap = " These estimates overlap and should not be added together." if len(stakes) > 1 else ""
        hold_stake = f" Monthly revenue at stake if the effects persist: {listed}.{overlap}"
    else:
        hold_stake = ""
    options.append(DecisionOption(
        id="hold_and_monitor", kind="hold",
        title="Hold, and keep watching the flagged segments" if causes else "Hold, and gather more evidence",
        rationale=(
            "Change nothing yet and track orders weekly in the flagged segments, to see whether "
            "the drop persists before committing money or margin."
            if causes else
            "The evidence does not single out a cause. The safest move is to keep collecting data "
            "before acting on any one explanation."
        ),
        addresses=[e.id for e in causes], confidence="n/a",
        impact="No cost, but nothing recovers on its own." + hold_stake,
        assumptions=["The drop is not getting worse quickly enough to make waiting expensive."],
        risks=["Delay: the revenue gap stays open for as long as you wait."],
        impact_value=None,
    ))

    return DecisionSet(
        options=options, insufficient=not causes,
        unresolved=unresolved, not_supported=not_supported,
    )
