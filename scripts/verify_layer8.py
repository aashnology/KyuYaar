"""
Layer 8 check: does the recommendation gate and rank correctly across the
scenarios that already have known ground truth?

Runs `recommend()` on all four Layer 7 scenarios, once under the placeholder
assumptions and once under generous ones, and prints what it decides. Run:
    python scripts/verify_layer8.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_data, scenario_dir  # noqa: E402
from decisions import build_options  # noqa: E402
from orchestrator import investigate  # noqa: E402
from recommend import project_all, recommend  # noqa: E402
from scenario import Assumptions  # noqa: E402
from synthetic import SCENARIOS  # noqa: E402
from toolkit import Toolkit  # noqa: E402

QUESTION = "Revenue dropped last month. Why?"
RUNS = {
    "placeholder assumptions": None,
    "generous (100% recovered, 12mo, no lag)": Assumptions(
        recovery_share=1.0, horizon_months=12, lag_months=0,
    ),
}


def check(name):
    orders, marketing = load_data(scenario_dir(name))
    inv = list(investigate(QUESTION, Toolkit(orders, marketing)))[-1].data["investigation"]
    ds = build_options(inv.evidence)
    print(f"\n{'=' * 78}\n{name}: {SCENARIOS[name].title}\n{'=' * 78}")
    if ds.insufficient:
        print("Evidence stage: no supported cause -> no options beyond hold.")
    for label, assumptions in RUNS.items():
        scenarios = project_all(ds, inv.evidence, orders, (lambda o: assumptions) if assumptions else None)
        rec = recommend(ds, inv.evidence, scenarios)
        print(f"\n  [{label}]  status = {rec.status}")
        print(f"  {rec.sentence}")
        for note in rec.notes:
            print(f"    note: {note}")
        for a in rec.assessments.values():
            extra = f" adjusted={a.adjusted_impact:+,.0f}" if a.adjusted_impact is not None else ""
            print(f"    {a.option_id:32} {a.outcome:10} impact={a.impact}{extra}")


def main():
    ok = True
    for name in SCENARIOS:
        check(name)
    # The two scenarios with no injected cause must never end up recommending
    # anything, under either set of assumptions.
    for name in ("demand_shock", "flat"):
        orders, marketing = load_data(scenario_dir(name))
        inv = list(investigate(QUESTION, Toolkit(orders, marketing)))[-1].data["investigation"]
        ds = build_options(inv.evidence)
        for assumptions in RUNS.values():
            scenarios = project_all(ds, inv.evidence, orders, (lambda o: assumptions) if assumptions else None)
            rec = recommend(ds, inv.evidence, scenarios)
            ok &= rec.recommended_id is None and rec.ranking == []
    print("\n" + "=" * 78)
    print("No-cause scenarios never forced a recommendation:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
