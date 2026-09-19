"""Markdown decision memo built from a finished investigation."""

from narration import label


def _fmt(x):
    return "-" if x is None else (f"{x:,.1f}" if isinstance(x, float) else str(x))


def build_report(inv, decision_set, chosen_id=None, note="") -> str:
    lines = [
        "# KyuYaar decision memo", "",
        f"**Question:** {inv.question}", "",
        "## Summary", "", inv.summary, "",
        "## Evidence chain", "",
        "| Strength | Type | Finding | Sample | Source |",
        "|---|---|---|---|---|",
    ]
    order = {"strong": 0, "moderate": 1, "weak": 2}
    for ev in sorted(inv.evidence, key=lambda e: (order[e.strength], e.evidence_type)):
        prov = ev.details.get("provenance", {})
        src = f"`{prov.get('tool', '?')}` in {prov.get('source', '?')}"
        lines.append(
            f"| {ev.strength} | {ev.evidence_type} | {ev.hypothesis} | {ev.sample_size} | {src} |"
        )
    lines += ["", "## Options considered", ""]
    for opt in decision_set.options:
        mark = " (chosen)" if opt.id == chosen_id else ""
        lines += [
            f"### {opt.title}{mark}", "",
            opt.rationale, "",
            f"- **Confidence in the evidence behind it:** {opt.confidence}",
            f"- **Expected impact:** {opt.impact}",
            "- **Assumes:** " + "; ".join(opt.assumptions),
            "- **Risks:** " + "; ".join(opt.risks), "",
        ]
    if decision_set.not_supported:
        lines += ["## Tested and not supported", ""] + [f"- {x}" for x in decision_set.not_supported] + [""]
    lines += ["## Still unresolved", ""] + [f"- {x}" for x in decision_set.unresolved] + [""]

    lines += ["## Decision", ""]
    chosen = next((o for o in decision_set.options if o.id == chosen_id), None)
    if chosen:
        lines.append(f"Chosen by the person responsible: **{chosen.title}**")
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
