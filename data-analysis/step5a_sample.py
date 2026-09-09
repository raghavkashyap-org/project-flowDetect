"""
STEP 5A - STRATIFIED 100k SAMPLE (one more streaming pass, RAM-safe)
Takes an evenly-spaced (systematic) sample per label so rare attacks survive
and time spread is preserved. Also finishes the exact label-conflict check:
rows whose feature-hash duplicated in Step 3B are re-examined for >1 label.
Usage:
    python step5a_sample.py [--file merged_data.csv] [--total 100000]
Inputs:
    reports/label_distribution.csv, reports/label_mapping.csv,
    logs/dup_hashes.npy (if global dedup ran in Step 3B)
Outputs:
    data/sample_100k.parquet (+ .csv twin for quick peeking)
    reports/label_conflicts.csv (dup feature-hash with >1 label, if any)
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="merged_data.csv")
ap.add_argument("--total", type=int, default=100000)
args = ap.parse_args()
FILE, TOTAL = args.file, args.total

os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

for f in ["reports/label_distribution.csv", "reports/label_mapping.csv"]:
    if not os.path.exists(f):
        raise SystemExit(f"ERROR: {f} missing. Run Step 4 first.")


def find_header(path, max_scan=200):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                break
            if line.startswith("Dst Port,"):
                return i
    raise SystemExit("ERROR: header not found.")


dist = pd.read_csv("reports/label_distribution.csv")
counts = dict(zip(dist["label"].astype(str), dist["count"].astype(int)))
grand = sum(counts.values())
take_total = min(TOTAL, grand)

# ---- quotas: proportional, rare labels get at least min(count,1000) ----
quotas, taken, seen_c = {}, {}, Counter()
if grand <= TOTAL:
    quotas = dict(counts)
else:
    for lab, c in counts.items():
        quotas[lab] = max(min(c, 1000), round(TOTAL * c / grand))
        quotas[lab] = min(quotas[lab], c)
    diff = TOTAL - sum(quotas.values())
    big = max(quotas, key=quotas.get)  # usually Benign absorbs the rounding
    quotas[big] = min(counts[big], quotas[big] + diff)
    if sum(quotas.values()) < TOTAL:  # tiny-file edge: top up anywhere possible
        for lab in sorted(counts, key=counts.get, reverse=True):
            room = counts[lab] - quotas[lab]
            add = min(room, TOTAL - sum(quotas.values()))
            quotas[lab] += add
            if sum(quotas.values()) >= TOTAL:
                break
taken = Counter()
print(f"Rows available: {grand:,} -> sampling {sum(quotas.values()):,}")
for lab in sorted(quotas, key=quotas.get, reverse=True):
    print(f"  quota {lab:24s} {quotas[lab]:,} of {counts[lab]:,}")

mapdf = pd.read_csv("reports/label_mapping.csv")
mapdict = dict(zip(mapdf["raw_label"].astype(str),
                   mapdf["canonical"].astype(str)))
for k in list(mapdict):
    mapdict.setdefault(k.strip(), mapdict[k])

dupset = None
if os.path.exists("logs/dup_hashes.npy"):
    dupset = set(int(x) for x in np.load("logs/dup_hashes.npy").tolist())
    print(f"Conflict check armed on {len(dupset):,} duplicated hashes.")
else:
    print("NOTE: logs/dup_hashes.npy missing (dedup was off) — "
          "conflict check skipped.")
conf = Counter()
missing_lab = 0

hidx = find_header(FILE)
reader = pd.read_csv(FILE, skiprows=hidx, header=0, chunksize=200000,
                     engine="c", on_bad_lines="skip", encoding="utf-8",
                     encoding_errors="replace", keep_default_na=True,
                     skipinitialspace=True, low_memory=False)
SPARSE_MIN = 73  # 83 cols - 10; ragged junk rows arrive NaN-padded
ragged_dropped = 0
kept, done = [], False
pbar = tqdm(unit=" rows", desc="Sampling")
for chunk in reader:
    chunk.columns = chunk.columns.str.strip()
    m = chunk["Dst Port"].astype(str).str.strip() == "Dst Port" \
        if "Dst Port" in chunk else False
    if not isinstance(m, bool) and bool(m.any()):
        chunk = chunk[~m]
    chunk = chunk.dropna(how="all")
    sparse = chunk.count(axis=1) < SPARSE_MIN
    if bool(sparse.any()):
        ragged_dropped += int(sparse.sum())
        chunk = chunk[~sparse]
    if len(chunk) == 0:
        continue
    labs_raw = chunk["Label"].astype(str).str.strip()
    missing_lab += int(chunk["Label"].isna().sum())
    labs = labs_raw.map(lambda x: mapdict.get(x, x))
    mask = np.zeros(len(chunk), dtype=bool)
    for i, lb in enumerate(labs.tolist()):
        if quotas.get(lb, 0) == 0:
            continue
        seen_c[lb] += 1
        want = (seen_c[lb] * quotas[lb]) // counts[lb]
        if want > taken[lb]:
            taken[lb] += 1
            mask[i] = True
    if mask.any():
        sub = chunk[mask].copy()
        sub["Label"] = labs[mask].values  # store canonical label
        kept.append(sub)
    if dupset:
        h = pd.util.hash_pandas_object(
            chunk.drop(columns=["Label"]), index=False).to_numpy()
        for hv, lb in zip(h, labs.tolist()):
            if int(hv) in dupset:
                conf[(int(hv), lb)] += 1
    pbar.update(len(chunk))
    if all(taken[lb] >= quotas[lb] for lb in quotas):
        done = True
        break
pbar.close()

sample = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
sample.to_parquet("data/sample_100k.parquet", index=False)
sample.to_csv("data/sample_100k.csv", index=False)
print(f"\nSample saved: {len(sample):,} rows "
      f"({'all quotas filled' if done else 'quotas below — see table'})")
print("quota vs taken:")
for lab in sorted(quotas, key=quotas.get, reverse=True):
    print(f"  {lab:24s} quota={quotas[lab]:,} taken={taken[lab]:,}")
print(f"rows skipped (missing label): {missing_lab:,}")
print(f"ragged sparse rows dropped: {ragged_dropped:,}")

if dupset:
    by_hash = {}
    for (hv, lb), n in conf.items():
        by_hash.setdefault(hv, {"labels": set(), "n": 0})
        by_hash[hv]["labels"].add(lb)
        by_hash[hv]["n"] += n
    conflicts = {hv: v for hv, v in by_hash.items() if len(v["labels"]) > 1}
    pd.DataFrame(
        [{"dup_hash": hv, "n_labels": len(v["labels"]),
          "labels": "|".join(sorted(v["labels"])), "occurrences": v["n"]}
         for hv, v in sorted(conflicts.items())],
        columns=["dup_hash", "n_labels", "labels", "occurrences"]).to_csv(
        "reports/label_conflicts.csv", index=False)
    print(f"label conflicts (same flow, >1 label): {len(conflicts)} "
          f"hashes; details in reports/label_conflicts.csv")
print("NEXT: python step5b_visuals.py")
