"""
Synthetic dataset generator for KyuYaar.

The scenarios (a known injected cause, or none) and the generator itself live in
src/synthetic.py. This script writes them to disk.

  python scripts/generate_data.py                       # default scenario -> data/
  python scripts/generate_data.py --scenario channel_loss
  python scripts/generate_data.py --all                 # default + every other scenario
  python scripts/generate_data.py some/folder           # default scenario -> some/folder

Every scenario is seeded, so the same command always writes the same files.
Non-default scenarios go to data/scenarios/<name>/ unless a folder is given.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import scenario_dir  # noqa: E402
from synthetic import CAUSE_DATE, END_DATE, SCENARIOS, START_DATE, get_scenario, write_scenario  # noqa: E402


def _write(name, out_dir):
    tables = write_scenario(name, out_dir)
    sc = get_scenario(name)
    print(f"[{name}] {sc.title}")
    for table, df in tables.items():
        print(f"  {table + ':':<11}{len(df):>6} rows -> {Path(out_dir) / (table + '.csv')}")
    causes = ", ".join(sc.expected_causes) if sc.expected_causes else "none (the correct answer is 'insufficient data')"
    print(f"  date range {START_DATE} to {END_DATE}; effects start {CAUSE_DATE}; expected causes: {causes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", nargs="?", help="folder to write to (default: the scenario's own folder)")
    parser.add_argument("--scenario", default="default", choices=sorted(SCENARIOS))
    parser.add_argument("--all", action="store_true", help="write every scenario to its own folder")
    args = parser.parse_args()

    if args.all:
        if args.out_dir:
            parser.error("--all writes to each scenario's own folder; drop the folder argument")
        for name in SCENARIOS:
            _write(name, scenario_dir(name))
        return
    _write(args.scenario, args.out_dir or scenario_dir(args.scenario))


if __name__ == "__main__":
    main()
