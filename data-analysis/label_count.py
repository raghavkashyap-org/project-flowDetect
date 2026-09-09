"""
LABEL COUNT - count every label over all 16M rows (RAM-safe for 12GB)
Reads ONLY the Label column in chunks. Nothing else is loaded into memory:
  500k-row chunk x 1 short-string column = ~30-50 MB. Totally safe.
Usage:
    python label_count.py [--file merged_data.csv] [--chunksize 500000]
Output:
    reports/label_counts.csv   (raw_label, count, pct)
    + printed raw table, whitespace/case audit, timing, RAM peak
"""
import argparse
import os
import time
from collections import Counter

import pandas as pd
import psutil
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="merged_data.csv")
ap.add_argument("--chunksize", type=int, default=500000)
args = ap.parse_args()
FILE, CHUNK = args.file, args.chunksize
os.makedirs("reports", exist_ok=True)


def find_header(path, max_scan=200):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                break
            if line.startswith("Dst Port,"):
                return i
    raise SystemExit("ERROR: header not found in " + path)


hidx = find_header(FILE)
print(f"File: {FILE} ({os.path.getsize(FILE) / 1e9:.2f} GB) | "
      f"chunksize: {CHUNK:,} | reading Label column only")

counts = Counter()
missing = 0
mid_headers = 0
total = 0
peak_ram = 0.0
proc = psutil.Process(os.getpid())
t0 = time.time()

reader = pd.read_csv(FILE, skiprows=hidx, header=0, usecols=["Label"],
                     chunksize=CHUNK, engine="c", on_bad_lines="skip",
                     encoding="utf-8", encoding_errors="replace",
                     keep_default_na=True, low_memory=False)
pbar = tqdm(unit=" rows", desc="Counting labels")
for chunk in reader:
    labs = chunk["Label"]
    mid = (labs.astype(str).str.strip() == "Label").sum()
    mid_headers += int(mid)
    if mid:
        labs = labs[labs.astype(str).str.strip() != "Label"]
    missing += int(labs.isna().sum())
    valid = labs.dropna().astype(str)
    if len(valid):
        counts.update(valid.value_counts().to_dict())
    total += len(chunk)
    peak_ram = max(peak_ram, proc.memory_info().rss / 1e9)
    pbar.update(len(chunk))
pbar.close()

secs = time.time() - t0
rows = pd.DataFrame(counts.most_common(), columns=["raw_label", "count"])
rows["pct"] = (100 * rows["count"] / max(rows["count"].sum(), 1)).round(4)
rows.to_csv("reports/label_counts.csv", index=False)

print(f"\n==== LABELS over {total:,} parsed rows "
      f"({secs:.0f}s, {total / max(secs, 0.01):,.0f} rows/s) ====")
print(rows.to_string(index=False))
print(f"\nmissing labels: {missing:,} | mid-file headers skipped: {mid_headers:,}")
print(f"peak RAM: {peak_ram:.2f} GB | distinct raw labels: {len(counts)}")

# whitespace/case audit: same label written differently?
audit = rows.copy()
audit["stripped"] = audit["raw_label"].str.strip()
audit["key"] = audit["stripped"].str.lower()
print("\nLabels differing only by space/case (must merge before training):")
found = False
for _, g in audit.groupby("key"):
    forms = sorted(g["raw_label"].unique().tolist())
    if len(forms) > 1:
        found = True
        print(f"  {forms}  (combined {int(g['count'].sum()):,})")
if not found:
    print("  none — labels already clean")
print("Saved reports/label_counts.csv")
