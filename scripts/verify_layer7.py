"""
Layer 7 check: is the system tuned to one dataset?

Runs the offline investigation over several random draws of every scenario and
reports, against each scenario's known ground truth:

  - how often each injected cause is recovered
  - how often anything else is called supported (false causes)
  - on scenarios with no cause, how often the decision layer ends in
    "insufficient data" instead of an action

Then feeds deliberately messy uploads through the validator and prints what it
says. Run:  python scripts/verify_layer7.py [draws]     (default 10; ~1 minute)
"""

import dataclasses
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from data_loader import DEFAULT_DATA_DIR, enrich  # noqa: E402
from decisions import build_options  # noqa: E402
from orchestrator import investigate  # noqa: E402
from synthetic import SCENARIOS, generate  # noqa: E402
from toolkit import Toolkit  # noqa: E402
from validation import TABLES, load_upload  # noqa: E402

FIRST_SEED = 1000
QUESTION = "Revenue dropped last month. Why?"


def one_draw(job):
    name, seed = job
    tables = generate(dataclasses.replace(SCENARIOS[name], seed=seed))
    tables["orders"]["order_date"] = pd.to_datetime(tables["orders"]["order_date"])
    tables["marketing"]["date"] = pd.to_datetime(tables["marketing"]["date"])
    orders, marketing = enrich(tables["orders"], tables["products"], tables["customers"], tables["marketing"])
    inv = list(investigate(QUESTION, Toolkit(orders, marketing)))[-1].data["investigation"]
    supported = sorted(e.id for e in inv.evidence if e.evidence_type == "statistical" and e.strength != "weak")
    base = next(e for e in inv.evidence if e.id == "obs_baseline_trend")
    return {
        "name": name, "seed": seed, "supported": supported,
        "insufficient": build_options(inv.evidence).insufficient,
        "revenue_change": base.value,
    }


def sweep(draws):
    jobs = [(name, FIRST_SEED + i) for name in SCENARIOS for i in range(draws)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one_draw, jobs))

    print("=" * 78)
    print(f"Ground-truth check over {draws} random draws per scenario (seeds {FIRST_SEED}..{FIRST_SEED + draws - 1})")
    print("=" * 78)
    all_pass = True
    for name, sc in SCENARIOS.items():
        rows = [r for r in results if r["name"] == name]
        expected = set(sc.expected_causes)
        print(f"\n{name}: {sc.title}")
        for cause in sorted(expected):
            hit = sum(cause in r["supported"] for r in rows)
            print(f"  {cause:<26} recovered in {hit}/{draws} draws")
        false = [(r["seed"], sorted(set(r["supported"]) - expected)) for r in rows if set(r["supported"]) - expected]
        print(f"  false causes (supported but not injected): {len(false)}/{draws} draws {false if false else ''}")
        if not expected:
            ins = sum(r["insufficient"] for r in rows)
            print(f"  ended in 'insufficient data': {ins}/{draws} draws")
            all_pass &= ins == draws
        changes = sorted(r["revenue_change"] for r in rows)
        print(f"  revenue vs. baseline across draws: {changes[0]:+.1f}% to {changes[-1]:+.1f}% (median {changes[len(changes) // 2]:+.1f}%)")
        all_pass &= not false
    print("\nNo false causes on any scenario, and every no-cause draw ended in 'insufficient data':",
          "PASS" if all_pass else "FAIL (see above)")
    return all_pass


def messy_uploads():
    print()
    print("=" * 78)
    print("Bring-your-own-CSV: what the validator says about messy files")
    print("=" * 78)
    raw = {t: pd.read_csv(DEFAULT_DATA_DIR / f"{t}.csv", dtype=str) for t in TABLES}

    def to_bytes(frame):
        return frame.to_csv(index=False).encode()

    def attempt(title, **overrides):
        sources = {t: to_bytes(overrides.get(t, raw[t])) if not isinstance(overrides.get(t, raw[t]), (bytes, type(None))) else overrides.get(t, raw[t]) for t in TABLES}
        result = load_upload(sources)
        print(f"\n{title}\n  -> {'accepted' if result.ok else 'REJECTED'}")
        for m in result.errors:
            print(f"     error:   {m}")
        for m in result.warnings:
            print(f"     warning: {m}")

    attempt("clean export")
    attempt("revenue column named 'sales', order_date missing", orders=raw["orders"].rename(columns={"revenue": "sales", "order_date": "when"}))
    orders = raw["orders"].copy()
    orders["revenue"] = orders["revenue"].map(lambda v: f"${float(v):,.2f}")
    orders.loc[:3, "revenue"] = "unknown"
    attempt("revenue as '$1,234.50' text, four unreadable values", orders=orders)
    dates = pd.to_datetime(raw["orders"]["order_date"])
    attempt("export cut off mid-month", orders=raw["orders"][dates <= "2026-08-14"])
    attempt("only two months of orders",
            orders=raw["orders"][dates < "2025-11-01"],
            marketing=raw["marketing"][pd.to_datetime(raw["marketing"]["date"]) < "2025-11-01"])
    attempt("products.csv missing", products=None)


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ok = sweep(draws)
    messy_uploads()
    sys.exit(0 if ok else 1)
