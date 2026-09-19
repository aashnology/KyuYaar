"""
Numeric guardrail for LLM-written text.

The model is told to quote figures exactly as tool results give them and to
introduce no other numerals. This module enforces that: any number in the
narration that cannot be matched, at the precision it is written, to a number
in the evidence gathered so far marks the narration as unsupported.

It checks numbers, not claims. A model can still misstate what a correct
number means, which is why strengths and caveats are also always shown
straight from the Evidence objects in the UI rather than only through prose.
"""

import re
from dataclasses import dataclass, field

_NUMBER = re.compile(r"(?<![\w.])[-+\u2212]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _parse(token: str):
    cleaned = token.replace(",", "").replace("\u2212", "-")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    mantissa = re.split(r"[eE]", cleaned)[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return abs(value), decimals


def numbers_in(text: str):
    """Yield (value, decimals, raw_token) for each number in `text`."""
    for match in _NUMBER.finditer(text or ""):
        parsed = _parse(match.group())
        if parsed:
            yield parsed[0], parsed[1], match.group()


def _collect(obj, out):
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(abs(float(obj)))
    elif isinstance(obj, str):
        for value, _, _ in numbers_in(obj):
            out.add(value)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, out)


def allowed_numbers(evidence) -> set:
    """Every figure the model is entitled to mention, given this evidence."""
    out = set()
    for ev in evidence:
        _collect(
            [ev.value, ev.baseline, ev.sample_size, ev.hypothesis, ev.caveats,
             ev.segment, {k: v for k, v in ev.details.items() if k != "provenance"}],
            out,
        )
    return out


@dataclass
class GuardrailResult:
    ok: bool
    unsupported: list = field(default_factory=list)


def check_text(text: str, evidence) -> GuardrailResult:
    allowed = allowed_numbers(evidence)
    unsupported = []
    for value, decimals, raw in numbers_in(text):
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(a - value) <= tolerance for a in allowed):
            unsupported.append(raw)
    return GuardrailResult(ok=not unsupported, unsupported=unsupported)
