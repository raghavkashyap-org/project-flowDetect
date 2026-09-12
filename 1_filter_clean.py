"""
Step 1  --  Filter + clean the raw CICIDS file.

What it does, per the audit rulebook:
  1. Keeps ONLY rows whose Label is in the KEEP set (the 7 chosen classes).
  2. Keeps ONLY the 32 Tier-1/2 feature columns (+ Label) in the OUTPUT matrix.
  3. Drops every row that is corrupt / un-satisfiable for training:
        - non-numeric feature, NaN, +inf / -inf (defensive; audit found none)
        - a negative value in a column that must be >= 0
          (the two Init*Win columns are EXEMPT because -1 is their valid sentinel)
  4. Removes exact duplicate ROWS (global) so no memorized copy leaks from
     train into validation/test.
  5. Writes a single compact Parquet matrix + a stats report.

DEDUP KEY (--dedup-key), important:
  * "full"     (DEFAULT, matches the audit): a row is a duplicate only if ALL
                78 measurement columns are identical (Label and the four
                Day/Hour/Minute/Second capture columns excluded). This is the
                audit's definition of an exact duplicate row -> ~437,928 drops.
  * "features" : duplicate if the 32 KEPT features are identical. MUCH more
                aggressive; it collapses genuinely distinct flows and can wipe
                out uniform classes (e.g. FTP-BruteForce 193,354 -> 53 rows).
                Only use it if you deliberately want one row per unique
                behavioural signature.

TWO-PASS streaming design (constant ~200 MB RAM, safe on a 12 GB machine):
  Pass 1 reads the file, decides keep/drop per row for labels + corruption,
         and accumulates one 64-bit hash per row.
  After pass 1 the global duplicate mask is computed with numpy (unique +
         first-occurrence), so only ~130 MB of hashes is held -- no Python set
         of 16 M objects.
  Pass 2 re-reads the file once and writes only the surviving rows to Parquet.

Usage (Linux/macOS):
    python3 1_filter_clean.py \
        --input      merged_data.csv \
        --labels-csv reports/label_mapping.csv \
        --outdir     data_out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import (FEATURES, KEEP_LABELS, LABEL_COL, SENTINEL_MINUS_ONE_COLS)

FEAT_FLOAT = [c for c in FEATURES]          # all 32
# columns that must NOT contain a negative value:
NONNEG_COLS = [c for c in FEATURES if c not in SENTINEL_MINUS_ONE_COLS]
# columns where -1 is a valid sentinel (anything below -1 is still corrupt):
SENTINEL_COLS = sorted(SENTINEL_MINUS_ONE_COLS)

# capture-metadata columns that are NOT part of a flow's identity
META_COLS = {"Day", "Hour", "Minute", "Second"}


def _norm(name: str) -> str:
    """Whitespace/BOM-insensitive compare for header resolution."""
    return "".join(str(name).split()).replace("\ufeff", "").casefold()


def resolve_columns(header: list[str]) -> tuple[dict, list[str]]:
    """Map requested feature names to their actual spelling in the file."""
    lookup = {_norm(h): h for h in header}
    resolved, missing = {}, []
    for feat in FEATURES + [LABEL_COL]:
        real = lookup.get(_norm(feat))
        if real is None:
            missing.append(feat)
        else:
            resolved[feat] = real
    return resolved, missing


def rowkey_columns(all_cols: list[str], actual_label: str) -> list[str]:
    """78 measurement columns = everything except Label + Day/Hour/Minute/Second."""
    norm_meta = {_norm(c) for c in META_COLS}
    out = []
    for c in all_cols:
        if c == actual_label:
            continue
        if _norm(c) in norm_meta:
            continue
        out.append(c)
    return out


def hash_rows(df: pd.DataFrame, key_cols: list[str],
              mode: str = "value") -> np.ndarray:
    """Deterministic uint64 hash per row over key_cols (vectorised, fast).

    mode="value" : parse the cells as float64 first. Two cells that differ only
                   in TEXT formatting ("2.5" vs "2.50" vs "2.500") hash the
                   SAME. This is the correct notion of redundancy for training
                   (numerically identical rows carry no extra information).
    mode="text"  : hash the raw text of the cells. Two cells differing only in
                   formatting hash DIFFERENTLY. This reproduces the audit's
                   "md5 of raw row bytes" definition exactly.
    """
    sub = df[key_cols]
    if mode == "value":
        try:
            sub = sub.astype(np.float64)
        except (ValueError, TypeError):
            pass
    return pd.util.hash_pandas_object(sub, index=False).to_numpy(dtype=np.uint64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="path to merged_data.csv (83-column CIC file)")
    ap.add_argument("--outdir", default="data_out")
    ap.add_argument("--labels-csv", default=None,
                    help="optional label_mapping.csv; KEEP set taken from its "
                         "'raw_label' column for rows whose canonical label is "
                         "in common.KEEP_LABELS.")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--dedup-key", choices=["full", "features"], default="full",
                    help="'full' (default, audit-consistent) = all 78 measurement "
                         "columns; 'features' = only the 32 kept features "
                         "(very aggressive, can destroy uniform classes).")
    ap.add_argument("--dedup-hash", choices=["value", "text"], default="value",
                    help="'value' (default) treats cells that differ only in "
                         "float formatting as identical (correct redundancy for "
                         "ML); 'text' reproduces the audit's md5-of-row-bytes "
                         "definition exactly.")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false", default=True)
    ap.add_argument("--exclude-labels", default="",
                    help="comma-separated labels to REMOVE from the KEEP set, "
                         "e.g. --exclude-labels 'FTP-BruteForce'")
    ap.add_argument("--min-class-rows", type=int, default=1000,
                    help="warn if any kept class has fewer than this many rows "
                         "AFTER dedup (default 1000). Such a class cannot be "
                         "trained or evaluated meaningfully.")
    ap.add_argument("--also-csv", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    # ---- header / column resolution --------------------------------------
    with open(args.input, "r", encoding="utf-8", errors="replace") as fh:
        header = next(fh).rstrip("\r\n").split(",")
    resolved, missing = resolve_columns(header)
    if missing:
        print("[FATAL] requested columns not found in header:")
        for m in missing:
            print(f"   MISSING:  {m!r}")
        print("Actual file header (first 40):", header[:40])
        return 2
    actual_features = [resolved[c] for c in FEATURES]
    actual_label = resolved[LABEL_COL]

    # key columns for duplicate detection
    if args.dedup_key == "full":
        key_cols = rowkey_columns(header, actual_label)
    else:
        key_cols = actual_features
    print(f"[ok] file columns detected: {len(header)} total; "
          f"{len(actual_features)} kept as features; label '{actual_label}'")
    print(f"[ok] dedup key = '{args.dedup_key}' over {len(key_cols)} columns; "
          f"hash = '{args.dedup_hash}'")

    # ---- derive the KEEP label set ---------------------------------------
    keep_labels = set(KEEP_LABELS)
    if args.labels_csv:
        if not os.path.exists(args.labels_csv):
            print(f"[FATAL] --labels-csv not found: {args.labels_csv}")
            return 2
        m = pd.read_csv(args.labels_csv)
        raw_col = "raw_label" if "raw_label" in m.columns else m.columns[0]
        can_col = "canonical" if "canonical" in m.columns else m.columns[0]
        keep_labels = set(
            m.loc[m[can_col].astype(str).str.strip().isin(KEEP_LABELS), raw_col]
             .astype(str).str.strip())
        if len(keep_labels) != len(KEEP_LABELS):
            print(f"[warn] matched {len(keep_labels)} of {len(KEEP_LABELS)} "
                  f"requested labels in {args.labels_csv}")
        print(f"[ok] KEEP labels resolved from {args.labels_csv}: "
              f"{sorted(keep_labels)}")

    if args.exclude_labels:
        drop = {x.strip() for x in args.exclude_labels.split(",") if x.strip()}
        unknown = drop - keep_labels
        if unknown:
            print(f"[warn] --exclude-labels names not in the KEEP set: "
                  f"{sorted(unknown)}")
        keep_labels -= drop
        print(f"[ok] after --exclude-labels, keeping: {sorted(keep_labels)}")

    # =====================================================================
    # PASS 1 -- decide keep/drop per row, accumulate hashes
    # =====================================================================
    print("\n--- pass 1/2: scan, filter, hash ---")
    read_kw = dict(chunksize=args.chunksize, encoding="utf-8", engine="c")
    if args.dedup_hash == "text":
        # Text-mode hashing must see the ORIGINAL text, so pandas must not
        # parse numbers for us. keep_default_na=False keeps "" as "" rather
        # than turning it into NaN, so the text hash is faithful.
        read_kw.update(dtype=str, keep_default_na=False)
        print("[ok] text-hash mode: reading all cells as strings in pass 1")
    hashes = []
    valid = []
    drop_counts = Counter()
    neg_by_col = Counter()
    kept_labels = Counter()          # filled in pass 2 (post-dedup, exact)
    chunk_sizes = []
    n_rows_raw = 0
    n_cand = 0

    reader = pd.read_csv(args.input, **read_kw)
    for i, chunk in enumerate(reader):
        chunk = chunk.rename(columns={actual_label: LABEL_COL})
        chunk = chunk.rename(columns=dict(zip(actual_features, FEATURES)))
        n_raw = len(chunk)
        chunk_sizes.append(n_raw)
        n_rows_raw += n_raw

        # 1) label filter
        lbl = chunk[LABEL_COL].astype(str).str.strip()
        ok = lbl.isin(keep_labels).to_numpy()
        drop_counts["label_not_selected"] += int((~ok).sum())

        # 2) corruption checks (only meaningful on selected-label rows)
        X = chunk[FEATURES].apply(pd.to_numeric, errors="coerce").astype(np.float64)
        bad = (X.isna() | np.isinf(X)).any(axis=1).to_numpy()
        drop_counts["nan_inf"] += int((bad & ok).sum())
        ok &= ~bad

        negmask = (X[NONNEG_COLS] < 0)
        if negmask.to_numpy().any():
            for c in NONNEG_COLS:
                cnt = int((negmask[c].to_numpy() & ok).sum())
                if cnt:
                    neg_by_col[c] += cnt
            n_neg = negmask.any(axis=1).to_numpy()
            drop_counts["illegal_negative"] += int((n_neg & ok).sum())
            ok &= ~n_neg

        sentmask = (X[SENTINEL_COLS] < -1.0)
        if sentmask.to_numpy().any():
            n_sent = sentmask.any(axis=1).to_numpy()
            drop_counts["sentinel_out_of_range"] += int((n_sent & ok).sum())
            ok &= ~n_sent

        # 3) hashes (needed for every row so pass-2 offsets line up)
        try:
            h = hash_rows(chunk, key_cols, mode=args.dedup_hash)
        except (KeyError, ValueError) as e:
            print(f"[FATAL] could not hash dedup key {key_cols[:5]}...: {e}")
            return 2
        hashes.append(h)
        valid.append(ok)

        if i % 5 == 0:
            n_cand += int(ok.sum())
            print(f"  chunk {i:>4} | raw={n_rows_raw:>12,} | "
                  f"candidates={n_cand:>12,} | "
                  f"{time.time()-t0:7.1f}s", flush=True)

    hashes = np.concatenate(hashes)
    valid = np.concatenate(valid)
    chunk_sizes = np.array(chunk_sizes, dtype=np.int64)
    assert len(hashes) == n_rows_raw, "hash/row count mismatch"

    # ---- global duplicate mask (keep FIRST occurrence) --------------------
    final_keep = valid.copy()
    if args.dedup:
        idx_valid = np.flatnonzero(valid)
        if len(idx_valid):
            hv = hashes[idx_valid]
            _, first_idx = np.unique(hv, return_index=True)
            dedup_keep = np.zeros(len(hv), dtype=bool)
            dedup_keep[first_idx] = True          # keep first occurrence only
            n_dup = int((~dedup_keep).sum())
            drop_counts["duplicate"] += n_dup
            final_keep[idx_valid] = dedup_keep
    print(f"\n[pass 1 done] raw={n_rows_raw:,}  "
          f"dropped={dict(drop_counts)}  kept={int(final_keep.sum()):,}  "
          f"({time.time()-t0:.1f}s)")

    # =====================================================================
    # PASS 2 -- re-read, write survivors
    # =====================================================================
    print("\n--- pass 2/2: write parquet ---")
    out_path = os.path.join(args.outdir, "filtered_clean.parquet")
    schema = pa.schema([pa.field(c, pa.float64()) for c in FEATURES]
                       + [pa.field(LABEL_COL, pa.string())])
    writer = pq.ParquetWriter(out_path, schema=schema)
    n_written = 0
    off = 0
    reader = pd.read_csv(args.input, **read_kw)  # same kwargs as pass 1
    for i, chunk in enumerate(reader):
        n_raw = len(chunk)
        if n_raw != chunk_sizes[i]:
            print(f"[FATAL] chunk {i} size drift {n_raw} != {chunk_sizes[i]}; "
                  "re-run with a stable file.")
            writer.close()
            return 2
        take = final_keep[off:off + n_raw]
        off += n_raw
        if not take.any():
            continue
        chunk = chunk.rename(columns={actual_label: LABEL_COL})
        chunk = chunk.rename(columns=dict(zip(actual_features, FEATURES)))
        out = chunk.loc[take, FEATURES].apply(pd.to_numeric, errors="coerce")
        out = out.astype(np.float64).copy()
        lab = chunk.loc[take, LABEL_COL].astype(str)
        out[LABEL_COL] = lab.to_numpy()
        kept_labels.update(lab.tolist())      # exact, post-dedup
        tbl = pa.Table.from_pandas(out.reset_index(drop=True), schema=schema,
                                   preserve_index=False)
        writer.write_table(tbl)
        n_written += int(take.sum())
        if i % 5 == 0:
            print(f"  chunk {i:>4} | written={n_written:>12,} | "
                  f"{time.time()-t0:7.1f}s", flush=True)
    writer.close()

    # ---- reconcile (fail loudly on any miscount) ---------------------------
    kept_labels = dict(sorted(kept_labels.items(), key=lambda kv: -kv[1]))
    if sum(kept_labels.values()) != n_written:
        print(f"[FATAL] label counts ({sum(kept_labels.values()):,}) != "
              f"rows written ({n_written:,})")
        return 2
    accounted = sum(drop_counts.values()) + n_written
    if accounted != n_rows_raw:
        print(f"[FATAL] row accounting broken: dropped+kept = {accounted:,} "
              f"but raw = {n_rows_raw:,} (off by {n_rows_raw-accounted:,})")
        return 2
    print(f"[ok] accounting OK: {n_rows_raw:,} = "
          f"{sum(drop_counts.values()):,} dropped + {n_written:,} kept")

    # ---- starved-class warning --------------------------------------------
    starved = {k: v for k, v in kept_labels.items() if v < args.min_class_rows}
    if starved:
        print("\n" + "!" * 72)
        print("!! WARNING: class(es) too small to train or evaluate:")
        for k, v in sorted(starved.items(), key=lambda kv: kv[1]):
            print(f"!!    {k:<24} {v:>10,} rows   (< --min-class-rows "
                  f"{args.min_class_rows})")
        print("!! These rows are almost certainly heavy duplication of a few")
        print("!! templates. A model will memorise them, and - because folds are")
        print("!! assigned by feature-content hash - the whole class lands in ONE")
        print("!! fold, so it is either absent from val/test or never seen in")
        print("!! training. Exclude it (--exclude-labels) or fold it into a")
        print("!! family class.")
        print("!" * 72 + "\n")

    # ---- report ------------------------------------------------------------
    report = {
        "raw_rows_read": n_rows_raw,
        "clean_rows_written": n_written,
        "dedup_key": args.dedup_key,
        "dedup_hash": args.dedup_hash,
        "dedup_key_ncols": len(key_cols),
        "dropped": dict(drop_counts),
        "negatives_by_column": dict(neg_by_col),
        "kept_labels": kept_labels,
        "features_kept": FEATURES,
        "output": out_path,
    }
    rep_path = os.path.join(args.outdir, "filter_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n==== FILTER COMPLETE ====")
    print(f"raw rows read     : {n_rows_raw:,}")
    print(f"clean rows written: {n_written:,}")
    print(f"drops: {dict(drop_counts)}")
    if neg_by_col:
        print(f"negatives by column: {dict(neg_by_col)}")
    print("kept label counts:")
    for k, v in kept_labels.items():
        print(f"   {k:<24} {v:>12,}")
    print(f"elapsed {time.time()-t0:.1f}s  ->  {rep_path}")

    if args.also_csv:
        import pyarrow.csv as pcs
        pcs.write_csv(pq.read_table(out_path),
                      os.path.join(args.outdir, "filtered_clean.csv"))
        print("wrote CSV twin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
