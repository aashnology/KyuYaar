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
from orchestrator import STANDARD_PLAN, investigate  # noqa: E402
from strength import BRIEF_NAME  # noqa: E402
from toolkit import Toolkit  # noqa: E402
from validation import TABLES, load_upload  # noqa: E402

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "olist_raw"
ADAPTED_DIR = REPO_ROOT / "data" / "olist_adapted"
QUESTION = "Revenue dropped last month. Why?"


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
    print(f"\nBring-your-own-CSV validator: {'ACCEPTED' if result.ok else 'REJECTED'}")
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
    inv = list(investigate(QUESTION, Toolkit(orders, marketing)))[-1].data["investigation"]
    for ev in inv.evidence:
        print(f"\n[{BRIEF_NAME[ev.strength]:>6} / {ev.strength:<8}] {ev.id}")
        print(f"  {ev.hypothesis}")
        if ev.caveats:
            print(f"  caveats: {'; '.join(ev.caveats)}")
    actionable = [e for e in inv.evidence if e.strength != "weak"]
    print(f"\n{len(actionable)}/{len(inv.evidence)} pieces of evidence came out Medium or High.")
    if not actionable:
        print("Nothing came out Medium/High on real data -- that is a valid, reportable result "
              "(see the mapping notes above for why), not a bug to work around.")
    return inv


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
        print("\nSkipping the investigation step: the validator rejected the adapted data (see errors above).")

    ok &= run_regression_suite()
    sys.exit(0 if ok else 1)
