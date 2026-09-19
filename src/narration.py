"""
Deterministic sentence generation from Evidence objects.

Two jobs: it is the offline-mode narrator, and it is the fallback whenever the
guardrail rejects text the LLM wrote. Everything here is built from fields on
the Evidence objects, so the numbers in it are correct by construction.
"""

_STRENGTH_WORD = {"strong": "Strong", "moderate": "Moderate", "weak": "Weak"}


def label(ev) -> str:
    """'region=North' -> 'North (region)'."""
    if not ev.segment or "=" not in ev.segment:
        return ev.segment or "overall"
    dim, value = ev.segment.split("=", 1)
    return f"{value} ({dim})"


def _notable(evidence):
    return [e for e in evidence if e.strength in ("strong", "moderate")]


def _first_caveat(ev):
    return ev.caveats[0][0].upper() + ev.caveats[0][1:] + "." if ev.caveats else ""


def narrate_step(tool, args, evidence) -> str:
    if not evidence:
        return "The tool returned no evidence."

    if tool == "baseline_trend":
        ev = evidence[0]
        text = (
            f"{ev.hypothesis}. This is a {ev.strength} signal, based on "
            f"{ev.sample_size} orders in the latest month."
        )
        if ev.caveats:
            text += " " + _first_caveat(ev)
        return text

    if tool == "segment_breakdown":
        dim = (args or {}).get("dimension", "segment")
        notable = _notable(evidence)
        if not notable:
            unstable = next((e for e in evidence if e.caveats and "unstable" in e.caveats[0]), None)
            base = f"No {dim} stands out beyond what its size alone would predict; the change looks spread out."
            return base + (" " + _first_caveat(unstable) if unstable else "")
        parts = [f"{_STRENGTH_WORD[e.strength]}: {e.hypothesis}." for e in notable]
        return (
            f"The change is concentrated in specific {dim} values. " + " ".join(parts)
            + " This shows where the change sits, not why it happened."
        )

    if tool in ("marketing_effect", "price_effect"):
        driver = "marketing spend" if tool == "marketing_effect" else "pricing"
        unit = "region" if tool == "marketing_effect" else "category"
        notable = _notable(evidence)
        if not notable:
            return (
                f"No {unit} shows a move in {driver} that lines up with an order "
                f"change, so {driver} is not supported as an explanation."
            )
        parts = []
        for e in notable:
            p = e.details.get("p_value_text")
            parts.append(
                f"{_STRENGTH_WORD[e.strength]} evidence: {e.hypothesis} "
                f"(p {p}, {e.sample_size} orders)."
            )
        plural = "regions" if unit == "region" else "categories"
        rest = f"The other {plural} show no matching move in {driver}."
        return " ".join(parts) + " " + rest

    return " ".join(e.hypothesis + "." for e in evidence[:3])


def build_summary(evidence) -> str:
    """Plain-language wrap-up of the whole evidence chain."""
    obs = next((e for e in evidence if e.evidence_type == "observation"), None)
    conc = [e for e in _notable(evidence) if e.evidence_type == "association"]
    causes = [e for e in _notable(evidence) if e.evidence_type == "statistical"]

    parts = []
    if obs:
        parts.append(obs.hypothesis + ".")
    if conc:
        names = ", ".join(label(e) for e in conc)
        parts.append(f"The change is concentrated in {names}.")
    if causes:
        described = "; ".join(f"{e.hypothesis} ({e.strength} evidence)" for e in causes)
        parts.append(f"Explanations the data supports: {described}.")
        parts.append(
            "These are associations, not proof of cause: the driver and the order "
            "change happened in the same month, so other simultaneous changes cannot "
            "be ruled out. Every other region and category tested shows no matching driver move."
        )
    else:
        parts.append(
            "The data does not support a specific cause. More data, or a controlled "
            "test, would be needed before acting on any single explanation."
        )
    return " ".join(parts)
