"""
Diagnostic: WHY does dedup drop far more rows than the audit's 437,928,
and why does FTP-BruteForce collapse to ~53 rows?

It answers three questions with numbers from YOUR file, on a sample
(default 3,000,000 rows -- fast, and enough to prove the point):

  Q1. Does the header contain duplicate column names (which would make a
      column selection silently return extra columns)?
  Q2. Per kept label: how many rows, how many DISTINCT rows under
        (a) parsed-value hash  (what 1_filter_clean.py uses)
        (b) raw-text hash      (what the audit's md5-of-row-bytes uses)
      If (a) << (b), the extra dedup is real value-level repetition that
      differs only in float FORMATTING.
  Q3. Is FTP-BruteForce really ~53 distinct numeric rows, or is that an
      artifact? Prints the top-5 most repeated FTP rows with their counts.

Usage:
    python3 diagnose_dedup.py --input merged_data.csv
    python3 diagnose_dedup.py --input merged_data.csv --max-rows 5000000
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from common import FEATURES, KEEP_LABELS, LABEL_COL, SENTINEL_MINUS_ONE_COLS

META = {"Day", "Hour", "Minute", "Second"}


def norm(s: str) -> str:
    return "".join(str(s).split()).replace("\ufeff", "").casefold()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--max-rows", type=int, default=3_000_000,
                    help="0 = whole file (slow).")
    ap.add_argument("--chunksize", type=int, default=50_000)
    ap.add_argument("--labels-csv", default=None)
    args = ap.parse_args()

    # ---------- header checks (Q1) --------------------------------------
    with open(args.input, "r", encoding="utf-8", errors="replace") as fh:
        header = next(fh).rstrip("\r\n").split(",")
    print("=" * 72)
    print(f"Q1. HEADER   total columns = {len(header)}")
    seen, dups = {}, []
    for h in header:
        k = norm(h)
        if k in seen:
            dups.append((seen[k], h))
        else:
            seen[k] = h
    if dups:
        print("    !! DUPLICATE COLUMN NAMES FOUND (normalised):")
        for a, b in dups:
            print(f"       {a!r}  and  {b!r}")
        print("    This makes df[cols] return extra columns -> hash over-merges.")
    else:
        print("    OK: all column names are unique (case/space-insensitive).")

    label_col = next(h for h in header if norm(h) == norm(LABEL_COL))
    key_cols = [h for h in header
                if h != label_col and norm(h) not in {norm(m) for m in META}]
    print(f"    dedup key columns (non-label, non-time) = {len(key_cols)}")

    keep = set(KEEP_LABELS)
    if args.labels_csv and os.path.exists(args.labels_csv):
        m = pd.read_csv(args.labels_csv)
        keep = set(m.loc[m["canonical"].astype(str).str.strip().isin(KEEP_LABELS),
                        "raw_label"].astype(str).str.strip())
    print(f"    keeping labels: {sorted(keep)}")

    # ---------- streaming pass (Q2/Q3) ----------------------------------
    print("=" * 72)
    print(f"Q2/Q3. scanning up to {args.max_rows or 'ALL'} rows "
          f"(chunksize={args.chunksize}) ...")
    per_label_total = Counter()
    value_sets = {}
    text_sets = {}
    ftp_counter: Counter = Counter()
    n_read = 0

    for chunk in pd.read_csv(args.input, usecols=[label_col] + key_cols,
                             chunksize=args.chunksize, encoding="utf-8",
                             dtype=str, engine="c", keep_default_na=False):
        lbl = chunk[label_col].str.strip()
        sel = lbl.isin(keep)
        if not sel.any():
            n_read += len(chunk)
            continue
        sub = chunk.loc[sel]
        lbl = lbl[sel]
        keys = sub[key_cols]

        # (b) raw-text hash -- isolates formatting differences
        th = pd.util.hash_pandas_object(keys, index=False).to_numpy(np.uint64)
        # (a) parsed-value hash -- what the pipeline uses
        vals = keys.apply(pd.to_numeric, errors="coerce").astype(np.float64)
        vh = pd.util.hash_pandas_object(vals, index=False).to_numpy(np.uint64)

        for lab in pd.unique(lbl):
            m = (lbl == lab).to_numpy()
            per_label_total[lab] += int(m.sum())
            value_sets.setdefault(lab, set()).update(vh[m].tolist())
            text_sets.setdefault(lab, set()).update(th[m].tolist())
            if lab == "FTP-BruteForce":
                ftp_counter.update(vh[m].tolist())

        n_read += len(chunk)
        if n_read % (args.chunksize * 10) == 0:
            print(f"    ... {n_read:,} rows scanned", flush=True)
        if args.max_rows and n_read >= args.max_rows:
            break

    print(f"    scanned {n_read:,} rows")
    print("=" * 72)
    print("Q2. DISTINCT ROWS PER LABEL  (value-hash vs raw-text hash)")
    print(f"{'label':<24}{'rows':>12}{'distinct_parsed':>17}"
          f"{'distinct_text':>15}{'collapse_x':>12}")
    for lab, n in per_label_total.most_common():
        dv = len(value_sets[lab])
        dt = len(text_sets[lab])
        ratio = (n / dv) if dv else float("nan")
        print(f"{lab:<24}{n:>12,}{dv:>17,}{dt:>15,}{ratio:>11.1f}x")

    print("=" * 72)
    print("Q3. MOST-REPEATED FTP-BruteForce SIGNATURES (parsed-value hash)")
    tot_ftp = sum(ftp_counter.values())
    print(f"    FTP rows scanned = {tot_ftp:,}  "
          f"distinct = {len(ftp_counter):,}")
    for i, (h, c) in enumerate(ftp_counter.most_common(5), 1):
        print(f"      #{i}: repeated {c:>7,}x  "
              f"({100*c/max(tot_ftp,1):5.1f}% of FTP rows)")

    print("=" * 72)
    print("READING:")
    print(" * If distinct_parsed << distinct_text -> the extra drops are real")
    print("   value-level repetition differing only in float formatting; the")
    print("   audit's byte-hash undercounted duplicates.")
    print(" * If FTP distinct_parsed is tiny while distinct_text is large, the")
    print("   class genuinely has few unique numeric profiles. No dedup mode")
    print("   can preserve it: it is a property of the data, and FTP-BruteForce")
    print("   is then NOT learnable as a separate class.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
