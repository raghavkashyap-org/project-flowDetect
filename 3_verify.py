"""
Step 3 -- Post-cleaning verification (streaming; safe on large files).

Prints a certificate over filtered_clean.parquet (or any splits file):
  * schema / shape / columns
  * completeness : 0 NaN, 0 inf expected
  * range        : 0 negatives expected (except -1 sentinel in Init*Win)
  * class balance table
  * UNIQUENESS   : per class, distinct 32-feature signatures vs row count.
                   This is the check that exposes classes which are really a
                   handful of templates duplicated thousands of times -- such a
                   class cannot be trained or evaluated meaningfully.
Aborts (exit 2) if the matrix is not train-ready.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import FEATURES, KEEP_LABELS, LABEL_COL, SENTINEL_MINUS_ONE_COLS

NONNEG = [c for c in FEATURES if c not in SENTINEL_MINUS_ONE_COLS]
SENT_COLS = sorted(SENTINEL_MINUS_ONE_COLS)

# a class whose rows / distinct-signatures exceeds this is almost certainly
# template duplication rather than genuinely varied traffic
SUSPECT_COLLAPSE = 100.0


def check(path: str, batch_size: int = 250_000,
          min_distinct: int = 1000) -> int:
    ok = True
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    n_rows = pf.metadata.num_rows
    print(f"file      : {path}")
    print(f"shape     : ({n_rows:,}, {len(names)})")
    print(f"columns   : {len(names)}  ({names[:3]} ... {names[-2:]})")

    missing = [c for c in FEATURES + [LABEL_COL] if c not in names]
    if missing:
        print("FAIL: missing required columns:", missing)
        return 2

    nan = inf = neg_non_sent = sent_bad = 0
    counts: dict[str, int] = {}
    sig_chunks: dict[str, list[np.ndarray]] = {}
    t0 = time.time()

    for batch in pf.iter_batches(batch_size=batch_size):
        pdf = batch.to_pandas()
        X = pdf[FEATURES].astype(np.float64)
        arr = X.to_numpy()

        nan += int(np.isnan(arr).sum())
        inf += int(np.isinf(arr).sum())
        neg_non_sent += int((X[NONNEG].to_numpy() < 0).sum())
        sent_bad += int((X[SENT_COLS].to_numpy() < -1).sum())

        labels = pdf[LABEL_COL].astype(str).to_numpy()
        for lab in np.unique(labels):
            m = labels == lab
            counts[lab] = counts.get(lab, 0) + int(m.sum())
            # value-based row signature (float64 bit semantics via pandas hash)
            h = pd_hash(X[m])
            sig_chunks.setdefault(lab, []).append(h)

    print("\n-- completeness / range --")
    print(f"NaN in any feature  : {nan:,}   {'OK' if nan == 0 else 'FAIL'}")
    print(f"+-inf in any feature: {inf:,}   {'OK' if inf == 0 else 'FAIL'}")
    print(f"negatives (non-sent): {neg_non_sent:,}   {'OK' if neg_non_sent == 0 else 'FAIL'}")
    print(f"sentinels < -1      : {sent_bad:,}   {'OK' if sent_bad == 0 else 'FAIL'}")
    ok &= (nan == 0 and inf == 0 and neg_non_sent == 0 and sent_bad == 0)

    total = sum(counts.values())
    print("\n-- class balance & uniqueness --")
    print(f"{'class':<24}{'rows':>13}{'distinct':>13}{'collapse':>10}  status")
    starved = []
    for lab in sorted(counts, key=lambda k: -counts[k]):
        h = np.concatenate(sig_chunks[lab])
        distinct = int(np.unique(h).size)
        ratio = counts[lab] / distinct if distinct else float("inf")
        status = "ok"
        if distinct < min_distinct:
            status = f"TOO FEW UNIQUE (< {min_distinct})"
            starved.append(lab)
            ok = False
        elif ratio > SUSPECT_COLLAPSE:
            status = f"suspect (> {SUSPECT_COLLAPSE:.0f}x repeat)"
            ok = False
        print(f"{lab:<24}{counts[lab]:>13,}{distinct:>13,}{ratio:>9.1f}x  {status}")
    print(f"{'TOTAL':<24}{total:>13,}")

    unexpected = set(counts) - KEEP_LABELS
    if unexpected:
        print("\nFAIL: unexpected label(s) present:", unexpected)
        ok = False
    if starved:
        print(f"\nFAIL: class(es) with < {min_distinct} distinct rows: {starved}")
        print("      A model will memorise these; folds land in a single split so")
        print("      the class is either absent from val/test or unseen in training.")

    print(f"\n[scan took {time.time()-t0:.1f}s]")
    return 0 if ok else 2


def pd_hash(X) -> np.ndarray:
    """Row-wise value hash of a DataFrame -> uint64 array (equal rows collide)."""
    import pandas as pd
    return pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--batch-size", type=int, default=250_000)
    ap.add_argument("--min-class-rows", type=int, default=1000,
                    help="fail if any class has fewer than this many "
                         "DISTINCT rows (default 1000)")
    args = ap.parse_args()
    rc = check(args.input, args.batch_size, args.min_class_rows)
    print("\nVERDICT:", "TRAIN-READY" if rc == 0 else "NOT READY")
    return rc


if __name__ == "__main__":
    sys.exit(main())
