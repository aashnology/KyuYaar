"""
Layer 9 check: does the pipeline still work on real, unaltered-content data?

Three things, in order:

  1. Adapt the raw Olist CSVs (see src/adapters/olist.py) into KyuYaar's
     canonical schema and run them through the real bring-your-own-CSV
     validator (src/validation.py) -- the same gate a person's own export
     would go through.
  2. Run the standard offline investigation plan (src/orchestrator.py's
     STANDARD_PLAN, no LLM) over the adapted data and print every piece of
     evidence with its strength, exactly as the app would show it.
  3. Re-run the full existing test suite, to show the adapter did not
     require touching src/ to make this work.

Run:  python scripts/verify_layer9.py [path-to-olist-csvs]
      (default path: data/olist_raw/, not committed -- see README)
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from adapters.olist import CORE_FILES, FUNNEL_FILES, adapt_all, write_canonical_csvs  # noqa: E402
from data_loader import load_data  # noqa: E402
from orchestrator import STANDARD_PLAN  # noqa: E402
from strength import BRIEF_NAME  # noqa: E402
from toolkit import Toolkit  # noqa: E402
from validation import TABLES, load_upload  # noqa: E402

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "olist_raw"
ADAPTED_DIR = REPO_ROOT / "data" / "olist_adapted"


def _check_raw_files_present(raw_dir):
    needed = list(CORE_FILES.values()) + list(FUNNEL_FILES.values())
    missing = [f for f in needed if not (raw_dir / f).exists()]
    if missing:
        print("=" * 78)
        print(f"No Olist data found in {raw_dir}")
        print("=" * 78)
        print(
            "\nDownload both Kaggle datasets and place every CSV directly in that folder:\n"
            "  - Brazilian E-Commerce Public Dataset by Olist "
            "(kaggle.com/datasets/olistbr/brazilian-ecommerce)\n"
            "  - Marketing Funnel by Olist "
            "(kaggle.com/datasets/olistbr/marketing-funnel-olist)\n"
            f"\nMissing file(s):\n  " + "\n  ".join(missing)
        )
        return False
    return True


def adapt_and_validate(raw_dir):
    print("=" * 78)
    print("Step 1: adapting raw Olist tables to the canonical schema")
    print("=" * 78)
    tables, report = adapt_all(raw_dir)
    write_canonical_csvs(tables, ADAPTED_DIR)
    print(f"\nWrote adapted CSVs to {ADAPTED_DIR}")
    print(f"Row counts: {report.counts}")
    print("\nMapping notes (the honest degradation list -- Layer 9 step 3):")
    for note in report.notes:
        print(f"  - {note}")

    sources = {name: ADAPTED_DIR / f"{name}.csv" for name in TABLES}
    result = load_upload(sources)
    print(f"\nBring-your-own-CSV validator (what a real person's own upload would face): "
          f"{'ACCEPTED' if result.ok else 'REJECTED'}")
    for m in result.errors:
        print(f"  error:   {m}")
    for m in result.warnings:
        print(f"  warning: {m}")
    if result.summary:
        print(f"  summary: {result.summary}")
    return result


def run_investigation(orders, marketing):
    print("\n" + "=" * 78)
    print("Step 2: standard investigation plan (offline, no LLM) on real data")
    print("=" * 78)
    # Run each tool independently rather than through orchestrator.investigate(),
    # which treats the plan as one sequence and aborts entirely on the first
    # tool that raises. marketing_effect/marketing_channel_analysis are expected
    # to fail outright here (marketing.csv has no rows past the funnel data's own
    # end date, already flagged above) -- that is a real, reportable gap, not a
    # reason to hide whether the core evidence tools worked.
    tk = Toolkit(orders, marketing)
    all_evidence = []
    for name, args in STANDARD_PLAN:
        try:
            all_evidence.extend(tk.run(name, args))
        except Exception as exc:
            print(f"\n[COULD NOT RUN] {name}{args or ''}: {type(exc).__name__}: {exc}")

    for ev in all_evidence:
        print(f"\n[{BRIEF_NAME[ev.strength]:>6} / {ev.strength:<8}] {ev.id}")
        print(f"  {ev.hypothesis}")
        if ev.caveats:
            print(f"  caveats: {'; '.join(ev.caveats)}")
    actionable = [e for e in all_evidence if e.strength != "weak"]
    print(f"\n{len(actionable)}/{len(all_evidence)} pieces of evidence came out Medium or High.")
    if not actionable:
        print("Nothing came out Medium/High on real data -- that is a valid, reportable result "
              "(see the mapping notes above for why), not a bug to work around.")
    return all_evidence


def run_regression_suite():
    print("\n" + "=" * 78)
    print("Step 3: full existing test suite (Layers 1-8), unmodified")
    print("=" * 78)
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q",
                              "--ignore=tests/test_olist_adapter.py"],
                             cwd=REPO_ROOT, capture_output=True, text=True)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
    print("Regression suite:", "PASS" if result.returncode == 0 else "FAIL")
    return result.returncode == 0


if __name__ == "__main__":
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RAW_DIR
    if not _check_raw_files_present(raw_dir):
        sys.exit(1)

    validation_result = adapt_and_validate(raw_dir)
    ok = validation_result.ok
    if ok:
        orders, marketing = validation_result.data
        run_investigation(orders, marketing)
    else:
        print("\nThe validator rejected the adapted data as a real upload would be (see errors above). "
              "Running the core pipeline directly instead, the same way the app loads its own shipped "
              "dataset (data_loader.enrich, bypassing the upload gate) -- this is what actually tests "
              "whether Layers 1-4 work on real data, independent of whether this particular pair of "
              "real datasets happens to line up well enough to pass the strict upload gate.")
        try:
            orders, marketing = load_data(ADAPTED_DIR)
            # Olist's own last month is famously sparse (the extract was pulled mid-month), which the
            # validator already flagged above as "partial last month" -- the same condition
            # validate_tables(drop_partial_last_month=True) exists to handle for any dataset, not an
            # Olist special case. Applying it here rather than leaving the partial month in.
            cutoff = orders["order_date"].max().to_period("M").start_time
            before = len(orders)
            orders = orders[orders["order_date"] < cutoff].reset_index(drop=True)
            marketing = marketing[marketing["date"] < cutoff].reset_index(drop=True)
            print(f"\nDropped the partial last month ({before - len(orders)} orders); investigating the "
                  f"last complete month instead, same as the app's own 'leave out the partial month' option.")
            run_investigation(orders, marketing)
        except Exception as exc:
            print(f"\nThe evidence engine itself could not run on this data ({type(exc).__name__}: {exc}).")
            ok = False

    ok &= run_regression_suite()
    sys.exit(0 if ok else 1)
