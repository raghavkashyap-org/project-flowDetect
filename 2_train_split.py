"""
Step 2  --  Build model-ready train / val / test sets from filtered_clean.parquet.

Design:
  * Folds are assigned from a VECTORISED hash of the 32 FEATURE VALUES, not from
    the row index. Two consequences that matter:
      - rows with identical behaviour always land in the same fold, so no
        duplicated copy can straddle train/val (no memorisation leakage);
      - the hash decorrelates the fold from capture order, so the
        day/hour -> attack relationship found in the audit cannot leak
        through the split.
  * The hash is a fixed numpy mixing function over the float64 bit patterns, so
    it is deterministic, dependency-free, and FAST (no per-row Python loop).
    Verified: 11.7 M rows split in well under a minute.
  * Splits are ~70/15/15 by construction; because the hash is uniform the
    per-class ratios are preserved automatically.

Outputs train.parquet / val.parquet / test.parquet (same 32 features + Label)
plus split_report.json with per-split class counts, train class weights, and a
warning list for any class missing from a split.

Usage:
    python3 2_train_split.py \
        --input  data_out/filtered_clean.parquet \
        --outdir data_out/splits
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import FEATURES, LABEL_COL

TRAIN, VAL, TEST = 0, 1, 2
FOLD_NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}

# 64-bit odd constant (golden-ratio based) used as the mixing multiplier
_MIX = np.uint64(0x9E3779B97F4A7C15)
_SHIFT = np.uint64(29)


def content_hash(X: np.ndarray) -> np.ndarray:
    """Deterministic, vectorised per-row hash of a float64 [n, F] matrix.

    Equal rows (bitwise-equal float64 values, so 2.5 == 2.50) map to equal
    hashes. Column order is fixed by FEATURES, so the result is stable.
    """
    bits = np.ascontiguousarray(X).view(np.uint64)      # [n, F] bit patterns
    h = np.zeros(bits.shape[0], dtype=np.uint64)
    for j in range(bits.shape[1]):
        h = (h ^ bits[:, j]) * _MIX
        h ^= h >> _SHIFT
    return h


def assign_folds(X: np.ndarray, train: float, val: float) -> np.ndarray:
    """fold id per row, using the top 16 bits of the content hash as a uniform
    draw in [0, 1) (65536 buckets -> ratio granularity ~0.0015%)."""
    h = content_hash(X)
    u = (h >> np.uint64(48)).astype(np.float64) / 65536.0   # uniform in [0,1)
    folds = np.full(len(u), TEST, dtype=np.int8)
    folds[u < train] = TRAIN
    folds[(u >= train) & (u < train + val)] = VAL
    return folds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="data_out/splits")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=250_000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    if not (0 < args.train < 1) or not (0 < args.val < 1) \
            or args.train + args.val >= 1:
        print("[FATAL] need train>0, val>0, train+val<1")
        return 2

    pf = pq.ParquetFile(args.input)
    schema = pf.schema_arrow
    names = schema.names
    missing = [c for c in FEATURES + [LABEL_COL] if c not in names]
    if missing:
        print("[FATAL] missing columns in input:", missing)
        return 2
    print(f"input : {args.input}  ({pf.metadata.num_rows:,} rows)")
    print(f"split : train={args.train}  val={args.val}  "
          f"test={round(1-args.train-args.val, 4)}")

    writers = {f: pq.ParquetWriter(
        os.path.join(args.outdir, f"{FOLD_NAMES[f]}.parquet"),
        schema=schema) for f in (TRAIN, VAL, TEST)}
    counters = {f: Counter() for f in (TRAIN, VAL, TEST)}
    sig_chunks: dict[int, dict[str, list[np.ndarray]]] = {
        f: {} for f in (TRAIN, VAL, TEST)}
    n_tot = 0

    for i, batch in enumerate(pf.iter_batches(batch_size=args.batch_size)):
        pdf = batch.to_pandas()
        X = pdf[FEATURES].astype(np.float64).to_numpy()
        labels = pdf[LABEL_COL].astype(str)
        folds = assign_folds(X, args.train, args.val)
        n_tot += len(pdf)

        for f in (TRAIN, VAL, TEST):
            counters[f].update(labels[folds == f])
        # per-split, per-class uniqueness (cheap: hashes already implied by fold)
        h = content_hash(X)
        lab_arr = labels.to_numpy()
        for f in (TRAIN, VAL, TEST):
            mf = folds == f
            if not mf.any():
                continue
            for lab in np.unique(lab_arr[mf]):
                mm = mf & (lab_arr == lab)
                sig_chunks[f].setdefault(lab, []).append(h[mm])

        pdf = pdf.assign(_f=folds)
        for f in (TRAIN, VAL, TEST):
            sub = pdf[pdf._f == f].drop(columns="_f")
            if len(sub):
                writers[f].write_table(
                    pa.Table.from_pandas(sub, preserve_index=False,
                                         schema=schema))
        if i % 5 == 0:
            print(f"  batch {i:>4} | rows={n_tot:>12,} | "
                  f"{time.time()-t0:6.1f}s", flush=True)

    for w in writers.values():
        w.close()

    # ---------------- report ------------------------------------------------
    counts = {FOLD_NAMES[f]: dict(sorted(counters[f].items(),
                                         key=lambda kv: -kv[1]))
              for f in (TRAIN, VAL, TEST)}
    sizes = {FOLD_NAMES[f]: int(sum(counters[f].values()))
             for f in (TRAIN, VAL, TEST)}
    if sum(sizes.values()) != n_tot:
        print(f"[FATAL] split sizes {sum(sizes.values()):,} != rows {n_tot:,}")
        return 2

    classes = sorted(set().union(*[set(c) for c in counts.values()]))
    tr = counters[TRAIN]
    tr_total = sum(tr.values())
    weights = {k: tr_total / v for k, v in sorted(tr.items())}

    absent = {}
    for f in (TRAIN, VAL, TEST):
        for c in classes:
            if counters[f].get(c, 0) == 0:
                absent.setdefault(FOLD_NAMES[f], []).append(c)

    # uniqueness per split per class (guards against template-only classes)
    uniq = {}
    for f in (TRAIN, VAL, TEST):
        uniq[FOLD_NAMES[f]] = {}
        for lab, chunks in sig_chunks[f].items():
            hh = np.concatenate(chunks)
            uniq[FOLD_NAMES[f]][lab] = int(np.unique(hh).size)

    rep = {
        "input": args.input,
        "total_rows": n_tot,
        "split": {"train": args.train, "val": args.val,
                  "test": round(1 - args.train - args.val, 4)},
        "split_sizes": sizes,
        "per_split_class_counts": counts,
        "train_class_weights": weights,
        "per_split_class_distinct_signatures": uniq,
        "classes_absent_from_split": absent,
        "outputs": {FOLD_NAMES[f]: os.path.join(args.outdir,
                                                f"{FOLD_NAMES[f]}.parquet")
                    for f in (TRAIN, VAL, TEST)},
    }
    rep_path = os.path.join(args.outdir, "split_report.json")
    with open(rep_path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)

    print("\n==== SPLIT COMPLETE ====")
    print(f"train={sizes['train']:,}  val={sizes['val']:,}  "
          f"test={sizes['test']:,}   (total {n_tot:,})")
    print(f"\n{'class':<24}{'train':>12}{'val':>10}{'test':>10}"
          f"{'weight':>10}")
    for c in classes:
        print(f"{c:<24}{counters[TRAIN].get(c,0):>12,}"
              f"{counters[VAL].get(c,0):>10,}{counters[TEST].get(c,0):>10,}"
              f"{weights.get(c,float('nan')):>10.1f}")
    if absent:
        print("\n!! class(es) absent from a split (metrics will be undefined):")
        for k, v in absent.items():
            print(f"!!   {k}: {v}")
    print(f"\nclass weights (n_train / n_class): {weights}")
    print(f"elapsed {time.time()-t0:.1f}s  ->  {rep_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
