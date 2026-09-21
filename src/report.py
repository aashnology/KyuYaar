"""Markdown decision memo built from a finished investigation."""

from narration import label
from strength import RANK


def _fmt(x):
    return "-" if x is None else (f"{x:,.1f}" if isinstance(x, float) else str(x))


def build_report(inv, decision_set, chosen_id=None, note="", scenarios=None, followups=None) -> str:
    scenarios = scenarios or {}
    chosen_here = any(o.id == chosen_id for o in decision_set.options)
    lines = [
        "# KyuYaar decision memo" if chosen_here else "# KyuYaar investigation report", "",
        f"**Question:** {inv.question}", "",
        "## Summary", "", inv.summary, "",
        "## Evidence chain", "",
        "| Strength | Type | Finding | Sample | Source |",
        "|---|---|---|---|---|",
    ]
    order = RANK
    for ev in sorted(inv.evidence, key=lambda e: (order[e.strength], e.evidence_type)):
        prov = ev.details.get("provenance", {})
        src = f"`{prov.get('tool', '?')}` in {prov.get('source', '?')}"
        lines.append(
            f"| {ev.strength} | {ev.evidence_type} | {ev.hypothesis} | {ev.sample_size} | {src} |"
        )
    if followups:
        lines += ["", "## Follow-up questions", ""]
        for qa in followups:
            how = ("written by the model, figures checked" if qa.source == "llm"
                   else "generated from the evidence")
            lines += [f"**Q:** {qa.question}", "", f"**A:** {qa.text}", ""]
            meta = f"_{how}_"
            if qa.evidence_ids:
                meta += "; evidence used: " + ", ".join(f"`{i}`" for i in qa.evidence_ids)
            elif not qa.covered:
                meta += "; not answerable from this investigation's evidence"
            lines += [meta, ""]
    lines += ["", "## Options considered", ""]
    for opt in decision_set.options:
        mark = " (chosen)" if opt.id == chosen_id else ""
        lines += [
            f"### {opt.title}{mark}", "",
            opt.rationale, "",
            f"- **Confidence in the evidence behind it:** {opt.confidence}",
            f"- **Expected impact:** {opt.impact}",
            "- **Assumes:** " + "; ".join(opt.assumptions),
            "- **Risks:** " + "; ".join(opt.risks),
        ]
        sc = scenarios.get(opt.id)
        if sc is not None:
            lines.append(f"- **Projected impact:** {sc.summary}")
            if sc.projectable and sc.kind != "hold":
                a = sc.assumptions
                lines.append(
                    f"- **Assumptions behind the projection (set by the person):** wins back "
                    f"{a.recovery_share:.0%} of the revenue at stake"
                    + (f", across {a.test_share:.0%} of the segment" if sc.kind == "test" else "")
                    + f"; {a.lag_months} months before it starts, {a.horizon_months} months ahead."
                )
        lines.append("")
    if decision_set.not_supported:
        lines += ["## Tested and not supported", ""] + [f"- {x}" for x in decision_set.not_supported] + [""]
    lines += ["## Still unresolved", ""] + [f"- {x}" for x in decision_set.unresolved] + [""]

    lines += ["## Decision", ""]
    chosen = next((o for o in decision_set.options if o.id == chosen_id), None)
    if chosen:
        lines.append(f"Chosen by the person responsible: **{chosen.title}**")
        sc = scenarios.get(chosen.id)
        if sc is not None and sc.steps:
            lines += [
                "", "How the projection for this option was worked out:", "",
                "| Step | Value | How |", "|---|---:|---|",
            ]
            for step in sc.steps:
                lines.append(f"| {step.label} | {step.shown} | {step.formula} |")
        if note.strip():
            lines += ["", f"Their note: {note.strip()}"]
    else:
        lines.append("No option chosen yet.")

    lines += [
        "", "---",
        f"Investigation mode: {inv.mode}"
        + (f" ({inv.provider}, {inv.model})" if inv.model else "")
        + f". Model-written passages blocked by the numeric guardrail: {len(inv.guardrail_blocks)}.",
        "Every figure above comes from a deterministic calculation in this repository; "
        "the model chose which analyses to run and wrote readouts that were checked against them.",
    ]
    return "\n".join(lines)
