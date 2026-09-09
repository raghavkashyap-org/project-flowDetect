"""
STEP 3B - CHUNKED PROFILING PASS (the main pass over all 16M rows)
Reads the file in chunks (default 200,000 rows). Per chunk it audits:
  missing | non-numeric | inf | negatives | running stats (mean/std/min/max) |
  raw label counts | port x label | day-hour x label | range checks |
  exact duplicates (64-bit row hash, features only so label conflicts are findable)
Usage:
    python step3b_profile.py [--file merged_data.csv] [--chunksize 200000]
                             [--no-global-dedup]
Outputs:
    reports/quality_table_83cols.csv  (one row per column: missing/nonnum/inf/neg/stats)
    reports/label_raw_counts.csv      (every distinct raw label + count)
    reports/port_label_top.csv        (top port x label combos)
    reports/time_label.csv            (day x hour x label counts)
    reports/duplicates_summary.csv    (parsed rows, dup rows, dup %, mid-headers dropped)
    reports/invalid_ranges.csv        (port/proto/hour/min/sec out-of-range counts)
    logs/pass1_chunks.csv             (per-chunk rows, seconds, RAM)
    logs/dup_hashes.npy               (hashes seen >1, used by Step 5A for conflict check)
RAM safety: chunk ~150-250MB at 200k rows. If process RAM passes 10GB, global
duplicate tracking switches off automatically (recorded in the summary).
"""
import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import psutil
from collections import Counter
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="merged_data.csv")
ap.add_argument("--chunksize", type=int, default=200000)
ap.add_argument("--no-global-dedup", action="store_true",
                help="skip global duplicate tracking (tight-RAM fallback)")
args = ap.parse_args()

FILE, CHUNK = args.file, args.chunksize
DEDUP_ON = not args.no_global_dedup

os.makedirs("reports", exist_ok=True)
os.makedirs("logs", exist_ok=True)

LABEL = "Label"
WIN_COLS = {"Init Fwd Win Byts", "Init Bwd Win Byts"}
RAM_GB_OFF = 10.0  # auto-disable global dedup above this


def find_header(path, max_scan=200):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                break
            if line.startswith("Dst Port,"):
                return i, [c.strip() for c in line.strip().split(",")]
    raise SystemExit("ERROR: header not found. Run Step 2 first.")


hidx, cols = find_header(FILE)
NUM_COLS = [c for c in cols if c != LABEL]
print(f"File: {FILE} | chunksize: {CHUNK:,} | numeric cols: {len(NUM_COLS)} "
      f"| global dedup: {'ON' if DEDUP_ON else 'OFF'}")

# ---- accumulators (small: only counters, never rows) ----
agg = {c: {"n": 0, "missing": 0, "nonnum": 0, "inf": 0, "neg": 0,
           "neg1": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}
       for c in NUM_COLS}
label_counter = Counter()
port_label = Counter()
time_label = Counter()
day_values = set()
bad = Counter()  # out-of-range + misc rejects
label_missing = 0
mid_dropped = 0
allna_dropped = 0
total_rows = 0
dup_rows = 0
SEEN = set()
DUP_HASHES = set()
dedup_events = []
chunk_logs = []
proc = psutil.Process(os.getpid())

reader = pd.read_csv(
    FILE, skiprows=hidx, header=0, chunksize=CHUNK, engine="c",
    on_bad_lines="skip", encoding="utf-8", encoding_errors="replace",
    na_values=["?", "-", "--"], keep_default_na=True,
    skipinitialspace=True, low_memory=False)
# NOTE: skiprows (not header=N) because pandas counts header= AFTER skipping
# blank lines, which misaligns when junk/blank lines precede the header.
SPARSE_MIN = len(cols) - 10  # rows with fewer non-NA values are ragged junk
ragged_dropped = 0
ragged_previews = []

t_all = time.time()
pbar = tqdm(unit=" rows", desc="Profiling")
for ci, chunk in enumerate(reader):
    t0 = time.time()
    chunk.columns = chunk.columns.str.strip()  # tolerate 'Second ' etc.

    # --- drop mid-file header rows + fully-empty rows ---
    m1 = chunk["Dst Port"].astype(str).str.strip() == "Dst Port" \
        if "Dst Port" in chunk else False
    m2 = chunk[LABEL].astype(str).str.strip() == LABEL \
        if LABEL in chunk else False
    mid = int(np.logical_or(m1, m2).sum()) if not isinstance(m1, bool) else 0
    if mid:
        chunk = chunk[~np.logical_or(m1, m2)]
    mid_dropped += mid
    before = len(chunk)
    chunk = chunk.dropna(how="all")
    allna_dropped += before - len(chunk)
    # Ragged short/junk lines arrive NaN-padded (C engine pads short rows);
    # their values sit in WRONG columns, so drop + count them (see Step 3A).
    sparse = chunk.count(axis=1) < SPARSE_MIN
    nsparse = int(sparse.sum())
    if nsparse:
        ragged_dropped += nsparse
        if len(ragged_previews) < 1000:
            for j in chunk[sparse].index[:1000 - len(ragged_previews)]:
                ragged_previews.append(
                    {"chunk": ci,
                     "preview": chunk.loc[j].astype(str).str.cat(sep=",")[:200]})
        chunk = chunk[~sparse]
    n = len(chunk)
    if n == 0:
        continue
    total_rows += n

    # --- labels (raw, to expose typos/spaces/case) ---
    lab_raw = chunk[LABEL]
    label_missing += int(lab_raw.isna().sum())
    labs = lab_raw.dropna().astype(str)
    if len(labs):
        label_counter.update(labs.value_counts().to_dict())
    lab_list = chunk[LABEL].astype(str).tolist()

    # --- joint counters (guarded size) ---
    if len(port_label) < 2_000_000 and "Dst Port" in chunk:
        ports = chunk["Dst Port"].astype(str).str.strip().tolist()
        port_label.update(zip(ports, lab_list))
    if len(time_label) < 2_000_000 and "Day" in chunk and "Hour" in chunk:
        dd = chunk["Day"].astype(str).str.strip().tolist()
        hh = chunk["Hour"].astype(str).str.strip().tolist()
        time_label.update(zip(dd, hh, lab_list))
    if "Day" in chunk and len(day_values) < 1000:
        day_values.update(chunk["Day"].astype(str).str.strip().unique().tolist())

    # --- duplicates (hash of FEATURES ONLY, so label conflicts stay findable) ---
    feats = chunk.drop(columns=[LABEL]) if LABEL in chunk else chunk
    h = pd.util.hash_pandas_object(feats, index=False).to_numpy()
    if DEDUP_ON:
        for hv, lb in zip(h, lab_list):
            hv = int(hv)
            if hv in SEEN:
                dup_rows += 1
                DUP_HASHES.add(hv)
            else:
                SEEN.add(hv)
    else:
        dup_rows += int(feats.duplicated(keep="first").sum())

    # --- per-column numeric audit (object cols convert, numeric cols direct) ---
    for c in NUM_COLS:
        if c not in chunk:
            continue
        a = agg[c]
        s_raw = chunk[c]
        miss = int(s_raw.isna().sum())
        a["missing"] += miss
        if pd.api.types.is_numeric_dtype(s_raw):
            s = s_raw
            nonnum = 0
        else:
            s = pd.to_numeric(s_raw, errors="coerce")
            nonnum = int(s.isna().sum()) - miss
        a["nonnum"] += nonnum
        a["inf"] += int(np.isinf(s).sum())
        finite = s[np.isfinite(s)]
        a["neg"] += int((finite < 0).sum())
        if c in WIN_COLS:
            a["neg1"] += int((finite == -1).sum())
        if len(finite):
            a["n"] += len(finite)
            a["sum"] += float(finite.sum())
            a["sumsq"] += float((finite ** 2).sum())
            mn, mx = float(finite.min()), float(finite.max())
            a["min"] = mn if a["min"] is None else min(a["min"], mn)
            a["max"] = mx if a["max"] is None else max(a["max"], mx)

    # --- range checks (finite values only; non-numeric already counted) ---
    def bad_range(col, lo, hi, name):
        if col not in chunk:
            return
        v = pd.to_numeric(chunk[col], errors="coerce")
        v = v[np.isfinite(v)]
        bad[name] += int(((v < lo) | (v > hi)).sum())

    bad_range("Dst Port", 0, 65535, "port_out_of_0_65535")
    bad_range("Protocol", 0, 255, "protocol_out_of_0_255")
    bad_range("Hour", 0, 23, "hour_out_of_0_23")
    bad_range("Minute", 0, 59, "minute_out_of_0_59")
    bad_range("Second", 0, 59, "second_out_of_0_59")

    # --- RAM guard ---
    rss = proc.memory_info().rss / 1e9
    secs = time.time() - t0
    chunk_logs.append({"chunk": ci, "rows": n, "seconds": round(secs, 1),
                       "ram_gb": round(rss, 2),
                       "cum_rows": total_rows, "cum_dups": dup_rows})
    if rss > RAM_GB_OFF and DEDUP_ON:
        DEDUP_ON = False
        SEEN.clear()
        gc.collect()
        dedup_events.append(f"global dedup auto-disabled at chunk {ci} "
                            f"(RAM {rss:.1f}GB); intra-chunk dups only after")
    pbar.update(n)
pbar.close()

# ---------------- save everything ----------------
qrows = []
for c in cols:
    if c == LABEL:
        qrows.append({"column": c, "parsed_rows": total_rows,
                      "missing": label_missing,
                      "missing_pct": round(100 * label_missing / max(total_rows, 1), 4),
                      "distinct_raw_values": len(label_counter),
                      "non_numeric": "", "inf": "", "neg": "", "neg1_sentinel": "",
                      "finite_n": "", "min": "", "mean": "", "std": "", "max": ""})
        continue
    a = agg.get(c, {"n": 0, "missing": 0, "nonnum": 0, "inf": 0, "neg": 0,
                    "neg1": 0, "sum": 0.0, "sumsq": 0.0, "min": None,
                    "max": None})
    mean = a["sum"] / a["n"] if a["n"] else ""
    var = (a["sumsq"] / a["n"] - mean ** 2) if a["n"] else ""
    std = float(np.sqrt(max(var, 0))) if var != "" else ""
    qrows.append({"column": c, "parsed_rows": total_rows,
                  "missing": a["missing"],
                  "missing_pct": round(100 * a["missing"] / max(total_rows, 1), 4),
                  "distinct_raw_values": "",
                  "non_numeric": a["nonnum"], "inf": a["inf"], "neg": a["neg"],
                  "neg1_sentinel": a["neg1"] if c in WIN_COLS else "",
                  "finite_n": a["n"], "min": a["min"],
                  "mean": (round(mean, 4) if mean != "" else ""),
                  "std": (round(std, 4) if std != "" else ""), "max": a["max"]})
pd.DataFrame(qrows).to_csv("reports/quality_table_83cols.csv", index=False)

pd.DataFrame(label_counter.most_common(),
             columns=["raw_label", "count"]).to_csv(
    "reports/label_raw_counts.csv", index=False)
pd.DataFrame([{"port": p, "raw_label": l, "count": n}
              for (p, l), n in port_label.most_common(300)]).to_csv(
    "reports/port_label_top.csv", index=False)
pd.DataFrame([{"day": d, "hour": h, "raw_label": l, "count": n}
              for (d, h, l), n in time_label.most_common(50000)]).to_csv(
    "reports/time_label.csv", index=False)
pd.DataFrame([{"metric": "parsed_rows", "value": total_rows},
              {"metric": "mid_file_header_rows_dropped", "value": mid_dropped},
              {"metric": "all_empty_rows_dropped", "value": allna_dropped},
              {"metric": "missing_label_rows", "value": label_missing},
              {"metric": "exact_duplicate_rows", "value": dup_rows},
              {"metric": "duplicate_pct",
               "value": round(100 * dup_rows / max(total_rows, 1), 4)},
              {"metric": "distinct_dup_hashes", "value": len(DUP_HASHES)},
              {"metric": "global_dedup_mode",
               "value": "full" if not dedup_events else
               "PARTIAL - " + "; ".join(dedup_events)},
              {"metric": "distinct_days_seen", "value": len(day_values)},
              {"metric": "minutes_total",
               "value": round((time.time() - t_all) / 60, 1)}]).to_csv(
    "reports/duplicates_summary.csv", index=False)
pd.DataFrame([{"check": k, "fail_count": v} for k, v in bad.items()]).to_csv(
    "reports/invalid_ranges.csv", index=False)
pd.DataFrame(chunk_logs).to_csv("logs/pass1_chunks.csv", index=False)
np.save("logs/dup_hashes.npy",
        np.array(sorted(DUP_HASHES), dtype=np.uint64))
with open("logs/day_values.txt", "w", encoding="utf-8") as o:
    o.write("\n".join(sorted(day_values)[:1000]))

print("\n==== STEP 3B DONE ====")
print(f"  parsed rows: {total_rows:,} in "
      f"{(time.time() - t_all) / 60:.1f} min")
print(f"  mid-file headers dropped: {mid_dropped:,}, "
      f"duplicates: {dup_rows:,} "
      f"({100 * dup_rows / max(total_rows, 1):.2f}%)")
print(f"  raw distinct labels: {len(label_counter)}, "
      f"missing labels: {label_missing:,}")
for e in dedup_events:
    print("  NOTE:", e)
print("NEXT: python step4_labels.py")
