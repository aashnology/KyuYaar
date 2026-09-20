"""
Layer 3 check: fewer orders vs. smaller orders.

The generator cuts North's orders through a marketing cut (order count only)
and raises Electronics prices by 10% (fewer orders, higher order value). A
correct decomposition should therefore show:

  overall      fewer orders, no clear change in order value
  North        fewer orders, no clear change in order value
  Electronics  fewer orders AND higher order value
  everything else: no clear change in either

It also runs the same decomposition over earlier months, where nothing was
injected, to see how often it flags something by chance.

Run:  python scripts/verify_layer3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_data
from decomposition import MIN_HISTORY, aov_volume_decomposition, signature_check
from effects import marketing_effect, price_effect

orders, marketing = load_data()

groups = [("overall", None), ("region", "region"), ("category", "category")]
evidence = []
print("=" * 78)
print("Orders vs. order value, latest month vs. the month before")
print("=" * 78)
for title, dim in groups:
    print(f"\n-- {title}")
    for ev in aov_volume_decomposition(orders, dim):
        evidence.append(ev)
        p = ev.details["p_value_text"]
        print(f"[{ev.strength.upper():>8}] p={p:<6} {ev.hypothesis}")

print()
print("=" * 78)
print("Does each supported cause show the pattern it predicts?")
print("=" * 78)
causes = [e for e in marketing_effect(orders, marketing) + price_effect(orders) if e.strength != "weak"]
checks = {}
for c in causes:
    r = signature_check(c, evidence)
    checks[c.id] = r["status"]
    print(f"{c.id}: {r['status']}\n    {r['text']}")

print()
print("=" * 78)
print("Ground-truth check")
print("=" * 78)
found = {e.id for e in evidence if e.strength != "weak"}
expected = {
    "decomp_orders_overall",
    "decomp_orders_region_North",
    "decomp_orders_category_Electronics",
    "decomp_aov_category_Electronics",
}
extra = found - expected
missing = expected - found
print("flagged, as expected:", sorted(found & expected))
print("flagged, unexpected: ", sorted(extra))
print("expected, missing:   ", sorted(missing))
patterns_ok = checks == {"stat_marketing_North": "consistent", "stat_price_Electronics": "consistent"}
print("PASS" if not missing and not extra and patterns_ok else "FAIL")

print()
print("=" * 78)
print("False alarms in earlier months (nothing was injected there)")
print("=" * 78)
period = orders["order_date"].dt.to_period("M")
months = sorted(period.unique())
total = flagged = strong = 0
for cutoff in months[MIN_HISTORY + 1:-1]:
    sub = orders[period <= cutoff]
    evs = [e for _, d in groups for e in aov_volume_decomposition(sub, d)]
    bad = [e for e in evs if e.strength != "weak"]
    total += len(evs)
    flagged += len(bad)
    strong += sum(e.strength == "strong" for e in bad)
    print(f"{cutoff}: {len(bad)} of {len(evs)} flagged", [e.id for e in bad])
print(f"\n{flagged} of {total} flagged ({strong} strong)")
