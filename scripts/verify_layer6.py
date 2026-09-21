"""
Layer 6 check: follow-up questions, trend charts and the decision log.

Follow-up answers come from the evidence only, so three things must hold:
  - no figure in an offline answer is missing from the evidence (numeric guardrail);
  - the two injected causes (North marketing, Electronics price) are what a "why"
    question surfaces, and every other region and category is reported as weak;
  - a question the investigation did not test is answered as not tested.

The trend series must reproduce the figures inside the findings they illustrate,
and the decision log must round-trip a decision without inventing an outcome.

Run:  python scripts/verify_layer6.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_data
from decision_log import DecisionLog, record_from_decision
from decisions import build_options
from followup import answer_question
from guardrail import check_text
from orchestrator import investigate
from scenario import run_scenario
from toolkit import Toolkit
from trends import driver_trend

orders, marketing = load_data()
inv = list(investigate("Revenue dropped last month. Why?", Toolkit(orders, marketing)))[-1].data["investigation"]
by_id = {e.id: e for e in inv.evidence}
failures = []


def check(ok, what):
    print(f"  [{'ok' if ok else 'FAIL'}] {what}")
    if not ok:
        failures.append(what)


print("=" * 78)
print("Follow-up questions, offline (answers built from the evidence fields)")
print("=" * 78)
questions = [
    "Why did revenue drop?", "Why did North drop?", "Is the loss in North mainly Paid?",
    "Did prices cause the Electronics decline?", "Are orders smaller or fewer?", "How sure are you?",
    "How much did revenue fall?", "Which region is hit hardest?", "What about customer segments?",
    "What should I do?",
]
for q in questions:
    a = answer_question(q, inv)
    ok = check_text(a.text, inv.evidence).ok and set(a.evidence_ids) <= set(by_id)
    print(f"  [{'ok' if ok else 'FAIL'}] {q:<44} -> {', '.join(a.evidence_ids) or ('not covered' if not a.covered else '-')}")
    if not ok:
        failures.append(q)

print()
why = answer_question("Why did revenue drop?", inv)
check("stat_marketing_North" in why.evidence_ids and "stat_price_Electronics" in why.evidence_ids,
      "a 'why' question surfaces both injected causes")

quiet = [("South", "region"), ("East", "region"), ("West", "region"),
         ("Apparel", "category"), ("Beauty", "category"), ("Home", "category"), ("Sports", "category")]
wrongly_backed = [n for n, _ in quiet
                  if any(w in answer_question(f"Is {n} a problem?", inv).text
                         for w in ("Strong evidence", "Moderate evidence"))]
check(not wrongly_backed, f"all {len(quiet)} untouched segments are reported as weak"
      + (f" (backed by mistake: {wrongly_backed})" if wrongly_backed else ""))
check(not answer_question("What about customer segments?", inv).covered, "an untested question is answered as not tested")
check(not answer_question("What should I do?", inv).covered, "advice is left to the Decision screen")

print()
print("=" * 78)
print("Trend series against the findings they illustrate (last month vs. the one before)")
print("=" * 78)
for ev_id in ("stat_marketing_North", "stat_price_Electronics"):
    ev = by_id[ev_id]
    t = driver_trend(ev, orders, marketing)
    move = lambda s: (s[-1] / s[-2] - 1) * 100
    print(f"  {ev_id}")
    for name, chart, figure in (
        ("driver", move(t.driver_segment), ev.details["driver_change_pct"]),
        ("orders", move(t.orders_segment), ev.details["raw_orders_change_pct"]),
        ("comparison orders", move(t.orders_comparison), ev.details["control_orders_change_pct"]),
    ):
        check(abs(chart - figure) < 0.3, f"{name}: chart {chart:+.1f}% vs. finding {figure:+.1f}%")

print()
print("=" * 78)
print("Decision log round trip")
print("=" * 78)
ds = build_options(inv.evidence)
option = next(o for o in ds.options if o.id == "restore_marketing_North")
scenario = run_scenario(option, inv.evidence, orders)
with tempfile.TemporaryDirectory() as tmp:
    log = DecisionLog(Path(tmp) / "decisions.jsonl")
    log.append(record_from_decision(inv, ds, option.id, "checking the round trip", scenario))
    (rec,) = DecisionLog(Path(tmp) / "decisions.jsonl").all()
    check(rec.outcome is None and rec.status == "awaiting_outcome", "a saved decision has no outcome")
    check(rec.projection["revenue_total"] == scenario.revenue_total, "the projection is stored as computed")
    check({e["id"] for e in rec.evidence} == {"obs_baseline_trend", "stat_marketing_North"},
          "the evidence behind the option is snapshotted")
    check(len(log.awaiting_outcome()) == 1, "it is listed as awaiting an outcome")

print()
print("All checks passed." if not failures else f"{len(failures)} check(s) FAILED: {failures}")
sys.exit(1 if failures else 0)
