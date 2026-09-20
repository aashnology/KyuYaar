"""
Layer 4 check: marketing by channel.

The generator cuts North's Paid spend by 45% in the final month. It lowers North's
order count in every channel, because the demand drop is applied to the region and
the channel recorded on each order is drawn independently of it. So a correct
channel analysis should show:

  North / Paid   spend cut, Paid orders down against comparison regions (supported)
                 the loss is REGION-WIDE: Organic, Email and Referral fell as much
                 cost per Paid order unchanged: less was bought, it did not perform worse
  every other channel and region: not supported

It then repeats the analysis over earlier months, where nothing was injected, to
count how often a channel is flagged by chance.

Run:  python scripts/verify_layer4.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from channels import marketing_channel_analysis
from data_loader import load_data

orders, marketing = load_data()
cells = marketing_channel_analysis(orders, marketing)

print("=" * 78)
print("Every paid channel in every region, latest month vs. the month before")
print("=" * 78)
for e in cells:
    d = e.details
    name = f"{d['channel']} in {d['region']}"
    move = f"spend {d['driver_change_pct']:+.1f}% (typical {d['driver_typical_change_pct']:+.1f}%)"
    effect = "n/a" if e.value is None else f"{e.value:+.1f}%"
    p = d["p_value_text"] if d["p_value_text"] is not None else "n/a"
    print(f"[{e.strength.upper():>8}] {name:<16} {move:<34} orders vs. comparison {effect:>7}  p={p:<6} n={e.sample_size}")

print()
print("=" * 78)
print("Detail for the channels whose spend moved")
print("=" * 78)
for e in cells:
    if not e.details["driver_moved"]:
        continue
    print(f"\n{e.id}: {e.hypothesis}")
    pat, eff = e.details["channel_pattern"], e.details["efficiency"]
    if pat:
        print(f"  specificity [{pat['status']}]: {pat['text']}")
    if eff:
        print(f"  efficiency  [{eff['verdict']}]: {eff['text']}")

print()
print("=" * 78)
print("Ground-truth check")
print("=" * 78)
supported = [e for e in cells if e.strength != "weak"]
north = next(e for e in cells if e.id == "chan_North_Paid")
checks = {
    "only North/Paid is supported": [e.id for e in supported] == ["chan_North_Paid"],
    "North/Paid loss is region-wide": north.details["channel_pattern"]["status"] == "region_wide",
    "cost per Paid order unchanged": north.details["efficiency"]["verdict"] == "unchanged",
}
quiet = [e for e in cells if not e.details["driver_moved"]]
chance = [e.id for e in quiet if e.details["p_value"] is not None and e.details["p_value"] < 0.05]
for name, ok in checks.items():
    print(f"{'ok  ' if ok else 'FAIL'} {name}")
print(f"\n{len(chance)} of {len(quiet)} cells with unchanged spend look significant on orders alone "
      f"({', '.join(chance) or 'none'}); the spend gate keeps all of them weak.")
print("PASS" if all(checks.values()) and all(e.strength == "weak" for e in quiet) else "FAIL")

print()
print("=" * 78)
print("False alarms in earlier months (nothing was injected there)")
print("=" * 78)
period = orders["order_date"].dt.to_period("M")
months = sorted(period.unique())
total = flagged = 0
for cutoff in months[1:-1]:
    sub = orders[period <= cutoff]
    mkt = marketing[marketing["date"].dt.to_period("M") <= cutoff]
    evs = marketing_channel_analysis(sub, mkt)
    bad = [e for e in evs if e.strength != "weak"]
    total += len(evs)
    flagged += len(bad)
    print(f"{cutoff}: {len(bad)} of {len(evs)} flagged", [e.id for e in bad])
print(f"\n{flagged} of {total} flagged")
