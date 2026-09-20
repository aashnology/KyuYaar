"""
Layer 5 check: the scenario engine.

Projections are arithmetic over the evidence and the person's assumptions, so
the check is that the arithmetic can be reproduced independently. This script
recomputes the North and Electronics projections straight from the raw CSVs,
without going through src/scenario.py, and compares. It then shows how the
result moves with the assumptions, since those are the person's to set.

Run:  python scripts/verify_layer5.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_data
from decisions import build_options
from orchestrator import investigate
from scenario import Assumptions, run_scenario
from toolkit import Toolkit

orders, marketing = load_data()
inv = list(investigate("Revenue dropped last month. Why?", Toolkit(orders, marketing)))[-1].data["investigation"]
options = {o.id: o for o in build_options(inv.evidence).options}
ev = {e.id: e for e in inv.evidence}

print("=" * 78)
print("Every option at the placeholder assumptions (50% won back, 1 month lag, 3 months)")
print("=" * 78)
for opt in options.values():
    sc = run_scenario(opt, inv.evidence, orders)
    print(f"\n[{opt.kind:>4}] {opt.title}\n       {sc.summary}")

# Independent recomputation from the raw CSVs.
raw = ROOT / "data"
o = pd.read_csv(raw / "orders.csv", parse_dates=["order_date"])
p = pd.read_csv(raw / "products.csv")
c = pd.read_csv(raw / "customers.csv")
j = o.merge(p, on="product_id").merge(c[["customer_id", "region"]], on="customer_id")
j["month"] = j["order_date"].dt.strftime("%Y-%m")
j["cogs"] = j["cost"] * j["quantity"]


def margin(mask, month):
    s = j[mask & (j["month"] == month)]
    return 1 - s["cogs"].sum() / s["revenue"].sum()


north, elec = ev["stat_marketing_North"].details, ev["stat_price_Electronics"].details
a = Assumptions(recovery_share=0.8, lag_months=1, horizon_months=6)

n_sc = run_scenario(options["restore_marketing_North"], inv.evidence, orders, a)
n_margin = margin(j["region"] == "North", north["prior_period"])
n_cut = north["spend_prior"] - north["spend_last"]
n_revenue = north["revenue_at_stake"] * 0.8 * 5
n_profit = n_revenue * n_margin - n_cut * 6

e_sc = run_scenario(options["rollback_price_Electronics"], inv.evidence, orders, a)
m0 = margin(j["category"] == "Electronics", elec["prior_period"])
m1 = margin(j["category"] == "Electronics", elec["period"])
e_revenue = elec["revenue_at_stake"] * 0.8 * 5
e_profit = (m0 * (elec["revenue_last"] + elec["revenue_at_stake"] * 0.8) - m1 * elec["revenue_last"]) * 5
e_be = (m1 - m0) * elec["revenue_last"] / (m0 * elec["revenue_at_stake"])

print()
print("=" * 78)
print("Independent recomputation (80% won back, 1 month lag, 6 months)")
print("=" * 78)
rows = [
    ("North revenue recovered", n_sc.revenue_total, n_revenue),
    ("North gross profit after added spend", n_sc.gross_profit_total, n_profit),
    ("Electronics revenue recovered", e_sc.revenue_total, e_revenue),
    ("Electronics gross profit change", e_sc.gross_profit_total, e_profit),
    ("Electronics break-even share", e_sc.break_even_share, e_be),
]
close = True
for name, engine, by_hand in rows:
    ok = abs(engine - by_hand) < 1e-6 * max(1.0, abs(by_hand))
    close &= ok
    print(f"{'ok  ' if ok else 'FAIL'} {name:<38} engine {engine:>12,.3f}   raw CSVs {by_hand:>12,.3f}")

print()
print("=" * 78)
print("Round trip: gross profit should be zero at each option's own break-even share")
print("=" * 78)
trip = True
for oid in ("restore_marketing_North", "rollback_price_Electronics"):
    be = run_scenario(options[oid], inv.evidence, orders).break_even_share
    if be is None or be > 1:
        print(f"{oid}: break-even at {be:.0%} of the revenue at stake, above what can come back; nothing to round-trip")
        continue
    gp = run_scenario(options[oid], inv.evidence, orders, Assumptions(recovery_share=be)).gross_profit_total
    ok = abs(gp) < 1e-6
    trip &= ok
    print(f"{'ok  ' if ok else 'FAIL'} {oid}: break-even {be:.1%}, gross profit there {gp:+.6f}")

print()
print("=" * 78)
print("Sensitivity: North restore, break-even share on gross profit (lag 1 month)")
print("=" * 78)
print(f"{'months ahead':>12} {'break-even':>12} {'GP at 100% won back':>22}")
for horizon in (3, 6, 12, 24):
    sc = run_scenario(options["restore_marketing_North"], inv.evidence, orders,
                      Assumptions(recovery_share=1.0, lag_months=1, horizon_months=horizon))
    print(f"{horizon:>12} {sc.break_even_share:>12.0%} {sc.gross_profit_total:>+22,.0f}")

print()
print("=" * 78)
print("Sensitivity: gross profit change over 3 months by share won back")
print("=" * 78)
print(f"{'won back':>9} {'North restore':>15} {'Electronics rollback':>21}")
for share in (0.0, 0.25, 0.5, 0.75, 1.0):
    aa = Assumptions(recovery_share=share)
    n = run_scenario(options["restore_marketing_North"], inv.evidence, orders, aa).gross_profit_total
    e = run_scenario(options["rollback_price_Electronics"], inv.evidence, orders, aa).gross_profit_total
    print(f"{share:>9.0%} {n:>+15,.0f} {e:>+21,.0f}")

print()
print("=" * 78)
print("Options the engine will not size")
print("=" * 78)
seq = run_scenario(options["sequence_fixes"], inv.evidence, orders)
hold = run_scenario(options["hold_and_monitor"], inv.evidence, orders)
print(f"sequence_fixes projectable: {seq.projectable}")
print(f"hold_and_monitor sums its stakes: {any('total' in s.label.lower() for s in hold.steps)}")

ok = close and trip and not seq.projectable and not any("total" in s.label.lower() for s in hold.steps)
print("\nPASS" if ok else "\nFAIL")
