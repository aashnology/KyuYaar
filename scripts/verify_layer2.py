"""
Layer 2 check against the known root cause in the synthetic dataset.

The generator injects two effects in the final month: a 45% cut to paid
marketing spend in North, and a 10% price increase on Electronics. A correct
investigation should call exactly those two supported and reject everything
else. Run:  python scripts/verify_layer2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_data
from decisions import build_options
from orchestrator import investigate
from toolkit import Toolkit

orders, marketing = load_data()
events = list(investigate("Revenue dropped last month. Why?", Toolkit(orders, marketing)))
inv = events[-1].data["investigation"]

print("=" * 70)
print("Evidence chain")
print("=" * 70)
for ev in inv.evidence:
    print(f"[{ev.strength.upper():>8}] ({ev.evidence_type:<11}) {ev.hypothesis}")

print()
print("=" * 70)
print("Ground-truth check")
print("=" * 70)
statistical = [e for e in inv.evidence if e.evidence_type == "statistical"]
supported = sorted(e.id for e in statistical if e.strength != "weak")
expected = ["stat_marketing_North", "stat_price_Electronics"]
print("supported causes:", supported)
print("expected causes: ", expected)
print("PASS" if supported == expected else "FAIL")

print()
print("=" * 70)
print("Decision options")
print("=" * 70)
for opt in build_options(inv.evidence).options:
    print(f"[{opt.kind:>4}] {opt.title}")
    print(f"       {opt.impact}")
