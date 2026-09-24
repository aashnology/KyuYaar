"""
Numeric guardrail for LLM-written text.

The model is told to quote figures exactly as tool results give them and to
introduce no other numerals. This module enforces that: any number in the
narration that cannot be matched, at the precision it is written, to a number
in the evidence gathered so far marks the narration as unsupported.

Direction is checked as well as magnitude, but only where a sign means
something. Fields that carry a direction (the headline value, the baseline,
percentage and percentage-point changes, dollar effects, confidence bounds,
test statistics) remember their sign, so a narration that writes "+12%" over
evidence of -12% is rejected. Counts, sample sizes, p-values, spend and other
levels have no direction to flip and are matched on magnitude alone.

The check fires on a sign the narration actually writes. "Orders fell 39%"
carries no sign token and matches on magnitude, so a wrong direction word
("rose 39%") is still outside what this module can see; that is one more
reason strengths and caveats are shown straight from the Evidence objects in
the UI rather than only through prose.

It checks numbers, not claims. A model can still misstate what a correct
number means.
"""

import re
from dataclasses import dataclass, field

_NUMBER = re.compile(r"(?<![\w.])[-+\u2212]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")

# Detail keys whose sign is meaningful name a percentage (anything containing
# "pct": orders_change_pct, segment_pct_change, ci_low_pct), a point or dollar
# effect (..._pts, ..._effect), or are one of the fixed names below. Anything
# else numeric in `details` is a count, a level or a p-value.
# tests/test_guardrail.py sweeps real evidence and fails if a key that ever goes
# negative falls outside this convention, so a new signed field cannot slip
# through as sign-blind unnoticed.
_SIGNED_SUFFIXES = ("_pts", "_effect")
_SIGNED_NAMES = frozenset({"value", "baseline", "t", "z"})


def is_signed_key(name) -> bool:
    """True when a number stored under `name` has a direction worth checking."""
    name = str(name).rsplit(".", 1)[-1]
    return name in _SIGNED_NAMES or "pct" in name or name.endswith(_SIGNED_SUFFIXES)


def _parse(token: str):
    """(magnitude, decimals, sign) for one numeric token; sign is 0 when the
    token carries none or the value is zero (zero has no direction)."""
    cleaned = token.replace(",", "").replace("\u2212", "-")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    mantissa = re.split(r"[eE]", cleaned)[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    sign = {"-": -1, "+": 1}.get(cleaned[:1], 0) if value != 0 else 0
    return abs(value), decimals, sign


def signed_numbers_in(text: str):
    """Yield (magnitude, decimals, sign, raw_token) for each number in `text`."""
    for match in _NUMBER.finditer(text or ""):
        parsed = _parse(match.group())
        if parsed:
            yield (*parsed, match.group())


def numbers_in(text: str):
    """Yield (magnitude, decimals, raw_token) for each number in `text`."""
    for magnitude, decimals, _, raw in signed_numbers_in(text):
        yield magnitude, decimals, raw


def _sign_of(number) -> int:
    number = float(number)          # numpy scalars do not support bool arithmetic
    return (number > 0) - (number < 0)


def _collect(obj, signed, neutral, is_signed=False):
    """Sort every number under `obj` into `signed` [(magnitude, sign)] or
    `neutral` {magnitude}. `is_signed` says whether the field it came from has
    a direction. Numbers inside prose count as signed only when the prose
    writes an explicit sign."""
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if is_signed:
            signed.append((abs(float(obj)), _sign_of(obj)))
        else:
            neutral.add(abs(float(obj)))
    elif isinstance(obj, str):
        for magnitude, _, sign, _ in signed_numbers_in(obj):
            if sign:
                signed.append((magnitude, sign))
            else:
                neutral.add(magnitude)
    elif isinstance(obj, dict):
        for key, v in obj.items():
            _collect(v, signed, neutral, is_signed_key(key))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, signed, neutral, is_signed)


def _allowed(evidence):
    signed, neutral = [], set()
    for ev in evidence:
        _collect(ev.value, signed, neutral, True)
        _collect(ev.baseline, signed, neutral, True)
        _collect(
            [ev.sample_size, ev.hypothesis, ev.caveats, ev.segment,
             {k: v for k, v in ev.details.items() if k != "provenance"}],
            signed, neutral,
        )
    return signed, neutral


def allowed_numbers(evidence) -> set:
    """Every magnitude the model is entitled to mention, given this evidence."""
    signed, neutral = _allowed(evidence)
    return {m for m, _ in signed} | neutral


@dataclass
class GuardrailResult:
    ok: bool
    unsupported: list = field(default_factory=list)
    # The subset of `unsupported` whose magnitude is in the evidence but whose
    # written sign contradicts it. Callers that excuse a figure the person
    # typed themselves must not excuse these: the person supplied a number,
    # not a direction.
    sign_flipped: list = field(default_factory=list)


def check_text(text: str, evidence) -> GuardrailResult:
    signed, neutral = _allowed(evidence)
    unsupported, flipped = [], []
    for magnitude, decimals, sign, raw in signed_numbers_in(text):
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        directional = [s for m, s in signed if abs(m - magnitude) <= tolerance]
        found = bool(directional) or any(abs(m - magnitude) <= tolerance for m in neutral)
        if not found:
            unsupported.append(raw)
        elif sign and directional and not any(s in (0, sign) for s in directional):
            # A directional field carries this magnitude and it points the other
            # way. Its magnitude repeated as plain prose elsewhere in the
            # evidence ("fell 12%") cannot vouch for the flipped sign.
            unsupported.append(raw)
            flipped.append(raw)
    return GuardrailResult(ok=not unsupported, unsupported=unsupported, sign_flipped=flipped)
