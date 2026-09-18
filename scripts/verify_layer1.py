import sys
sys.path.insert(0, "/home/claude/kyuyaar/src")

import pandas as pd
from evidence import baseline_trend, segment_breakdown

DATA_DIR = "/home/claude/kyuyaar/data"

orders = pd.read_csv(f"{DATA_DIR}/orders.csv", parse_dates=["order_date"])
products = pd.read_csv(f"{DATA_DIR}/products.csv")
customers = pd.read_csv(f"{DATA_DIR}/customers.csv")

enriched = (
    orders
    .merge(products[["product_id", "category"]], on="product_id")
    .merge(customers[["customer_id", "region"]], on="customer_id")
)


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
