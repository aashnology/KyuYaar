"""
Deterministic sentence generation from Evidence objects.

Two jobs: it is the offline-mode narrator, and it is the fallback whenever the
guardrail rejects text the LLM wrote. Everything here is built from fields on
the Evidence objects, so the numbers in it are correct by construction.
"""

from decomposition import signature_check

_STRENGTH_WORD = {"strong": "Strong", "moderate": "Moderate", "weak": "Weak"}


def label(ev) -> str:
    """'region=North' -> 'North (region)'; a channel cell reads 'Paid in North'."""
    if not ev.segment or "=" not in ev.segment:
        return ev.segment or "overall"
    if "|" in ev.segment:
        parts = dict(p.split("=", 1) for p in ev.segment.split("|"))
        return f"{parts.get('channel', '?')} in {parts.get('region', '?')}"
    dim, value = ev.segment.split("=", 1)
    return f"{value} ({dim})"


def metric_note(ev) -> str:
    """Which factor a decomposition finding is about; empty for other evidence."""
    if ev.evidence_type != "decomposition":
        return ""
    return "orders" if ev.metric == "orders_change_pct" else "order value"


def _move(ev):
    return f"{'down' if ev.value < 0 else 'up'} {abs(ev.value):.1f}%"


def _pairs(evidence):
    """{segment: (orders_evidence, aov_evidence)} from decomposition evidence."""
    by_seg = {}
    for e in evidence:
        if e.evidence_type != "decomposition" or e.value is None:
            continue
        by_seg.setdefault(e.segment, {})["orders" if e.metric == "orders_change_pct" else "aov"] = e
    return {seg: (v["orders"], v["aov"]) for seg, v in by_seg.items() if len(v) == 2}


def _split_sentence(name, orders, aov):
    """One sentence on whether the change is fewer orders, smaller orders, or both."""
    o_move, a_move = orders.strength != "weak", aov.strength != "weak"
    o, a = _move(orders), _move(aov)
    if o_move and not a_move:
        return (
            f"{name}: the change is in the number of orders ({o}; {orders.strength} evidence), "
            f"not in order size (average order value {a}; no clear evidence it moved)."
        )
    if a_move and not o_move:
        return (
            f"{name}: the change is in order size (average order value {a}; {aov.strength} evidence), "
            f"not in the number of orders ({o}; no clear evidence it moved)."
        )
    if o_move and a_move:
        opposite = (orders.value < 0) != (aov.value < 0)
        tail = " They move in opposite directions, so each partly offsets the other." if opposite else ""
        return (
            f"{name}: order count {o} ({orders.strength} evidence) and average order value "
            f"{a} ({aov.strength} evidence).{tail}"
        )
    return f"{name}: neither order count ({o}) nor average order value ({a}) shows a clear change."


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

    if tool == "aov_volume_decomposition":
        pairs = _pairs(evidence)
        dim = (args or {}).get("dimension")
        if dim is None:
            orders, aov = pairs.get(None, (None, None))
            if orders is None:
                return "The tool returned no evidence."
            return _split_sentence("Overall (vs. the prior month)", orders, aov)
        shown = [(seg, o, a) for seg, (o, a) in pairs.items() if o.strength != "weak" or a.strength != "weak"]
        if not shown:
            return (
                f"No {dim} shows a clear move in either order count or average order value, "
                f"so the split between fewer and smaller orders is not distinguishable from ordinary variation."
            )
        parts = [_split_sentence(label(o), o, a) for _, o, a in shown]
        rest = f"The other {'regions' if dim == 'region' else 'categories'} show no clear move in either."
        return " ".join(parts) + " " + rest

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

    if tool == "marketing_channel_analysis":
        notable = _notable(evidence)
        if not notable:
            return (
                "No paid channel shows a spend move that lines up with an order change in that "
                "channel, so channel-level marketing is not supported as an explanation."
            )
        return (
            " ".join(_channel_lines(notable))
            + " Every other channel and region shows no matching move in spend."
        )

    return " ".join(e.hypothesis + "." for e in evidence[:3])


def _channel_lines(cells):
    """Sentences for supported channel findings: the finding, whether it is
    concentrated in the channel or region-wide, and what happened to cost per order."""
    lines = []
    for e in cells:
        line = f"{e.hypothesis} ({e.strength} evidence; p {e.details.get('p_value_text')}, {e.sample_size} orders)."
        pattern, eff = e.details.get("channel_pattern"), e.details.get("efficiency")
        if pattern:
            line += " " + pattern["text"]
        if eff:
            line += " " + eff["text"]
        lines.append(line)
    return lines


def build_summary(evidence) -> str:
    """Plain-language wrap-up of the whole evidence chain."""
    obs = next((e for e in evidence if e.evidence_type == "observation"), None)
    conc = [e for e in _notable(evidence) if e.evidence_type == "association"]
    causes = [e for e in _notable(evidence) if e.evidence_type == "statistical"]
    channels = [e for e in _notable(evidence) if e.evidence_type == "channel"]

    parts = []
    if obs:
        parts.append(obs.hypothesis + ".")
    overall = _pairs(evidence).get(None)
    if overall:
        parts.append(_split_sentence("Overall (vs. the prior month)", *overall))
    if conc:
        names = ", ".join(label(e) for e in conc)
        parts.append(f"The change is concentrated in {names}.")
    if causes:
        bullets = "\n".join(f"- {e.hypothesis} ({e.strength} evidence)" for e in causes)
        parts.append("Explanations the data supports:\n" + bullets)
        checks = [signature_check(e, evidence) for e in causes]
        lines = [c["text"] for c in checks if c]
        if lines:
            parts.append("Does the order pattern match each explanation?\n" + "\n".join(f"- {t}" for t in lines))
        if channels:
            parts.append("Which channel?\n" + "\n".join(f"- {t}" for t in _channel_lines(channels)))
        parts.append(
            "These are associations, not proof of cause: the driver and the order "
            "change happened in the same month, so other simultaneous changes cannot "
            "be ruled out. Every other region and category tested shows no matching driver move."
        )
    else:
        if channels:
            parts.append("Which channel?\n" + "\n".join(f"- {t}" for t in _channel_lines(channels)))
        parts.append(
            "The data does not support a specific cause. More data, or a controlled "
            "test, would be needed before acting on any single explanation."
        )
    return "\n\n".join(parts)
