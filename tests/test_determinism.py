"""
Same question, same data: identical evidence.

The tools are pure calculations, so this should hold. These tests prove it end
to end (question in, evidence and options out) instead of assuming it, in
three ways: on repeat runs within a process, across processes with different
hash seeds (which reorders sets and dicts if anything depends on that), and
with different model behaviour on top (narration may vary; evidence may not).
"""

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_loader import load_data, scenario_dir
from decisions import build_options
from orchestrator import STANDARD_PLAN, investigate
from synthetic import SCENARIOS
from toolkit import Toolkit

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "Revenue dropped last month. Why did it happen, and what should I do?"


def _dump(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def fingerprint(inv):
    """Everything numeric or structural about an investigation, as one string."""
    options = build_options(inv.evidence)
    return _dump({
        "evidence": [dataclasses.asdict(e) for e in inv.evidence],
        "plan": [(s["tool"], s["args"]) for s in inv.steps],
        "options": [dataclasses.asdict(o) for o in options.options],
        "unresolved": options.unresolved,
        "not_supported": options.not_supported,
    })


def run(scenario="default"):
    orders, marketing = load_data(scenario_dir(scenario))          # a fresh load every time
    events = list(investigate(QUESTION, Toolkit(orders, marketing)))
    return events[-1].data["investigation"]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_repeat_runs_give_identical_evidence_plan_and_options(scenario):
    first, second = run(scenario), run(scenario)
    assert fingerprint(first) == fingerprint(second)
    # Offline narration is template text built from the evidence, so it repeats too.
    assert first.summary == second.summary
    assert [s["narration"] for s in first.steps] == [s["narration"] for s in second.steps]


def test_the_fingerprint_actually_detects_a_difference():
    a = run("default")
    b = run("channel_loss")
    assert fingerprint(a) != fingerprint(b)


def test_results_do_not_depend_on_python_hash_ordering():
    snippet = (
        "import sys, hashlib, dataclasses, json; sys.path.insert(0, 'src');"
        "from data_loader import load_data; from toolkit import Toolkit;"
        "from orchestrator import investigate; from decisions import build_options;"
        "o, m = load_data();"
        f"inv = list(investigate({QUESTION!r}, Toolkit(o, m)))[-1].data['investigation'];"
        "opts = build_options(inv.evidence);"
        "blob = json.dumps({'e': [dataclasses.asdict(e) for e in inv.evidence],"
        "'o': [dataclasses.asdict(x) for x in opts.options]}, sort_keys=True, default=str);"
        "print(hashlib.sha256(blob.encode()).hexdigest())"
    )
    digests = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT / "src")}
        out = subprocess.run([sys.executable, "-c", snippet], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True)
        digests.add(out.stdout.strip())
    assert len(digests) == 1


# --- a live model on top: its wording and tool ordering may vary, the evidence may not ---

def _call(i, name, **args):
    return SimpleNamespace(type="tool_use", id=f"tu_{i}", name=name, input=args)


def _turn(*blocks):
    return SimpleNamespace(content=list(blocks), stop_reason="end_turn")


class ScriptedModel:
    def __init__(self, turns):
        self.turns = list(turns)
        self.messages = SimpleNamespace(create=lambda **kw: self.turns.pop(0))


def live_run(order, summary):
    orders, marketing = load_data()
    calls = [_call(i, name, **args) for i, (name, args) in enumerate(order)]
    model = ScriptedModel([
        _turn(SimpleNamespace(type="text", text="Working through the standard checks."), *calls),
        _turn(SimpleNamespace(type="text", text=summary)),
    ])
    events = list(investigate(QUESTION, Toolkit(orders, marketing), client=model))
    return events[-1].data["investigation"]


def test_a_live_model_changes_the_wording_and_order_but_never_the_evidence():
    a = live_run(STANDARD_PLAN, "The evidence points to two supported explanations.")
    b = live_run(list(reversed(STANDARD_PLAN)), "Two explanations hold up; the rest do not.")

    assert a.mode == b.mode == "live"
    assert a.summary != b.summary                                              # wording varies
    assert [s["tool"] for s in a.steps] != [s["tool"] for s in b.steps]        # so does tool order

    by_id = lambda inv: {e.id: _dump(dataclasses.asdict(e)) for e in inv.evidence}
    assert by_id(a) == by_id(b)                                                # the evidence does not

    offline = run("default")
    assert by_id(a) == by_id(offline)                                          # and matches the offline run
