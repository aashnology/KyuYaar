import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_data
from evidence import baseline_trend, segment_breakdown

enriched, _ = load_data()


def show(evidence):
    print(f"[{evidence.strength.upper():>8}] {evidence.hypothesis}")
    print(f"           type={evidence.evidence_type}  n={evidence.sample_size}  "
          f"value={evidence.value}  baseline={evidence.baseline}")
    for c in evidence.caveats:
        print(f"           caveat: {c}")
    print()


print("=" * 70)
print("STEP 1: baseline_trend() -- did anything change overall?")
print("=" * 70)
trend = baseline_trend(enriched, metric_col="revenue")
show(trend)

print("=" * 70)
print("STEP 2: segment_breakdown() by region -- who's responsible?")
print("=" * 70)
for e in segment_breakdown(enriched, dimension_col="region", metric_col="revenue"):
    show(e)

print("=" * 70)
print("STEP 3: segment_breakdown() by category -- who's responsible?")
print("=" * 70)
for e in segment_breakdown(enriched, dimension_col="category", metric_col="revenue"):
    show(e)
