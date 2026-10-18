"""
Clean the FULL raw hospital discharge file once, from the command line.

The dashboard does this automatically the first time it starts, so running this script is OPTIONAL -
use it if you want to see progress in the terminal, change the row cap, or pre-build the cache.

Usage
-----
    python prepare_data.py                          # uses data/raw_sparcs.csv
    python prepare_data.py --input path/to/file.csv
    python prepare_data.py --max-rows 0             # keep EVERY valid row (big output!)
    python prepare_data.py --max-rows 300000        # random sample of ~300k rows

Outputs (in data/)
------------------
    inpatient_clean.csv.gz   cleaned + engineered data (read by app.py)
    cleaning_report.json     row counts for every cleaning step (shown in the app)
    clean_meta.json          remembers which raw file / row cap produced the cache
"""
from __future__ import annotations

import argparse
import os
import sys

from src import processing as P

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(HERE, "data", "raw_sparcs.csv"))
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--max-rows", type=int, default=500_000, help="random-sample cap (0 = keep all rows)")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()
    max_rows = args.max_rows or None

    try:
        df, report = P.build_clean_from_raw(
            args.input, max_rows=max_rows, chunksize=args.chunksize,
            progress=lambda f, m: print(f"[{f * 100:5.1f}%] {m}", flush=True))
    except P.MissingColumnsError as e:
        print("ERROR: could not find required columns:", e.missing)
        print("Columns in your file:", e.available)
        print("Add the right names to COLUMN_ALIASES in src/processing.py and re-run.")
        return 1

    ok = P.save_clean(df, report, args.data_dir, args.input, max_rows)
    out = os.path.join(args.data_dir, P.CLEAN_NAME)
    if not ok:
        print("ERROR: could not write to", args.data_dir)
        return 1
    size_mb = os.path.getsize(out) / 1e6
    print(f"\nSaved {len(df):,} rows -> {out} ({size_mb:.1f} MB)")
    for r in report:
        print(f"  - {r['step']}: {r['rows_before']:,} -> {r['rows_after']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
