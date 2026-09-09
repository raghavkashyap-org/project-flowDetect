"""
WHOLE-DATA VISUALS - exact charts over ALL 16M rows (no sampling)
Technique: stream the file in chunks and accumulate small exact aggregates:
  label counts | fixed log-bin histograms per (feature, label) |
  label x hour | label x port | range violations.
Nothing larger than one chunk (~100-200MB) is ever in RAM.
Usage:
    python whole_data_visuals.py [--file merged_data.csv] [--chunksize 500000]
Outputs:
    reports/whole_histograms.npz, whole_label_time.csv, whole_label_port.csv,
    whole_ranges.csv  (aggregates - re-plot without re-reading 5.8GB)
    figures/figW01_...figW07_*.png  (7 exact whole-data charts)
"""
import argparse
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style="whitegrid", palette="deep")

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="merged_data.csv")
ap.add_argument("--chunksize", type=int, default=500000)
args = ap.parse_args()
FILE, CHUNK = args.file, args.chunksize
os.makedirs("reports", exist_ok=True)
os.makedirs("figures", exist_ok=True)

COLS = ["Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts",
        "Tot Bwd Pkts", "TotLen Fwd Pkts", "Flow Pkts/s", "Flow IAT Mean",
        "Pkt Len Mean", "Label", "Day", "Hour", "Minute", "Second"]
HIST_FEATS = ["Flow Duration", "Flow Pkts/s", "Flow IAT Mean",
              "Pkt Len Mean", "TotLen Fwd Pkts", "Tot Fwd Pkts"]
BINS = np.logspace(-3, 9, 73)  # 72 log bins, 0.001 .. 1e9 (fixed, no pre-pass)
NB = len(BINS) - 1
I_UNDER, I_ZERO, I_NEG, I_NAN, I_INF, I_OVER = NB, NB + 1, NB + 2, NB + 3, \
    NB + 4, NB + 5
NW = NB + 6


def find_header(path, max_scan=200):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                break
            if line.startswith("Dst Port,"):
                return i
    raise SystemExit("ERROR: header not found in " + path)


# label mapping from Step 4 if it exists (merges typo variants), else strip()
mapdict = {}
if os.path.exists("reports/label_mapping.csv"):
    m = pd.read_csv("reports/label_mapping.csv")
    mapdict = dict(zip(m["raw_label"].astype(str), m["canonical"].astype(str)))
    print("Using reports/label_mapping.csv for canonical labels.")
else:
    print("NOTE: no label_mapping.csv (Step 4 not run) - using stripped labels.")


def canon(s):
    s = str(s).strip()
    return mapdict.get(s, s)


hidx = find_header(FILE)
print(f"File: {FILE} ({os.path.getsize(FILE) / 1e9:.2f} GB) | "
      f"chunksize: {CHUNK:,}")

label_counts = Counter()
hists = {}  # (feat, label) -> int array length NW
time_counts = Counter()
port_counts = Counter()
range_bad = Counter()
total = missing_lab = mid = invalid_time = 0
proc = psutil.Process(os.getpid())
peak = 0.0
t0 = time.time()

reader = pd.read_csv(FILE, skiprows=hidx, header=0, usecols=COLS,
                     chunksize=CHUNK, engine="c", on_bad_lines="skip",
                     encoding="utf-8", encoding_errors="replace",
                     keep_default_na=True, low_memory=False)
pbar = tqdm(unit=" rows", desc="Aggregating all rows")
for chunk in reader:
    chunk.columns = chunk.columns.str.strip()
    mh = ((chunk["Label"].astype(str).str.strip() == "Label")
          | (chunk["Dst Port"].astype(str).str.strip() == "Dst Port"))
    mid += int(mh.sum())
    if bool(mh.any()):
        chunk = chunk[~mh]
    miss = chunk["Label"].isna()
    missing_lab += int(miss.sum())
    chunk = chunk[~miss]
    if len(chunk) == 0:
        continue
    total += len(chunk)
    labs = chunk["Label"].astype(str).map(canon)
    label_counts.update(labs.value_counts().to_dict())
    lab_arr = labs.to_numpy()

    # --- fixed-bin histograms per feature (exact, chunked) ---
    for feat in HIST_FEATS:
        v = pd.to_numeric(chunk[feat], errors="coerce").to_numpy(dtype=float)
        cat = np.full(len(v), I_NAN, dtype=np.int64)
        finite = np.isfinite(v)
        cat[np.isinf(v)] = I_INF
        cat[finite & (v < 0)] = I_NEG
        cat[finite & (v == 0)] = I_ZERO
        pos = finite & (v > 0)
        cat[pos & (v < 1e-3)] = I_UNDER
        cat[pos & (v > 1e9)] = I_OVER
        midm = pos & (v >= 1e-3) & (v <= 1e9)
        if midm.any():
            cat[midm] = np.digitize(v[midm], BINS) - 1
        for lab in np.unique(lab_arr):
            key = (feat, lab)
            arr = hists.get(key)
            if arr is None:
                arr = np.zeros(NW, dtype=np.int64)
                hists[key] = arr
            arr += np.bincount(cat[lab_arr == lab], minlength=NW)

    # --- time + ports + ranges ---
    d = pd.to_numeric(chunk["Day"], errors="coerce")
    h = pd.to_numeric(chunk["Hour"], errors="coerce")
    ok_t = (np.isfinite(d) & np.isfinite(h) & (d >= 0) & (d <= 31)
            & (h >= 0) & (h <= 23)).to_numpy()
    invalid_time += int((~ok_t).sum())
    if ok_t.any():
        tt = (d[ok_t].astype(int) * 24 + h[ok_t].astype(int)).tolist()
        ll = lab_arr[ok_t].tolist()
        time_counts.update(zip(tt, ll))
    p = pd.to_numeric(chunk["Dst Port"], errors="coerce")
    pok = np.isfinite(p) & (p >= 0) & (p <= 65535)
    range_bad["port_invalid"] += int((~pok).sum())
    ps = chunk["Dst Port"].astype(str).str.strip().str.replace(r"\.0$", "",
                                                              regex=True)
    ps = ps.where(pok.to_numpy(), "INVALID")
    port_counts.update(zip(ps.tolist(), lab_arr.tolist()))
    for col, lo, hi, nm in [("Protocol", 0, 255, "protocol_invalid"),
                            ("Hour", 0, 23, "hour_invalid"),
                            ("Minute", 0, 59, "minute_invalid"),
                            ("Second", 0, 59, "second_invalid")]:
        vv = pd.to_numeric(chunk[col], errors="coerce")
        vv = vv[np.isfinite(vv)]
        range_bad[nm] += int(((vv < lo) | (vv > hi)).sum())

    peak = max(peak, proc.memory_info().rss / 1e9)
    pbar.update(len(chunk))
pbar.close()

secs = time.time() - t0
labels = sorted(label_counts)
print(f"\nAggregated {total:,} rows in {secs:.0f}s "
      f"({total / max(secs, 0.01):,.0f} rows/s, peak RAM {peak:.2f}GB)")
print(f"labels: {len(labels)} | missing labels: {missing_lab:,} | "
      f"mid-headers: {mid:,} | invalid time rows: {invalid_time:,}")

# ---- persist aggregates (re-plot later without re-reading 5.8GB) ----
H = np.zeros((len(HIST_FEATS), len(labels), NW), dtype=np.int64)
for i, f in enumerate(HIST_FEATS):
    for j, lab in enumerate(labels):
        if (f, lab) in hists:
            H[i, j] = hists[(f, lab)]
np.savez_compressed("reports/whole_histograms.npz", bins=BINS,
                    feats=np.array(HIST_FEATS), labels=np.array(labels),
                    hists=H,
                    counts=np.array([label_counts[l] for l in labels]))
pd.DataFrame([{"t": t, "label": l, "count": n}
              for (t, l), n in time_counts.items()]).to_csv(
    "reports/whole_label_time.csv", index=False)
pd.DataFrame([{"port": p, "label": l, "count": n}
              for (p, l), n in port_counts.most_common(500)]).to_csv(
    "reports/whole_label_port.csv", index=False)
pd.DataFrame([{"check": k, "fail_count": v}
              for k, v in range_bad.items()]).to_csv(
    "reports/whole_ranges.csv", index=False)

# ---- plots (all exact whole-data) ----
TAG = f"ALL {total:,} rows (exact, chunked - no sampling)"
ben = next((l for l in labels if l.lower() == "benign"), None)


def save(fig, name, suptitle=False):
    if suptitle:
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    else:
        fig.tight_layout()
    fig.savefig(f"figures/{name}", dpi=120)
    plt.close(fig)
    print(" ", name)


lc = pd.DataFrame(label_counts.most_common(), columns=["label", "count"])
fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(lc) + 2)))
ax.barh(lc["label"][::-1], lc["count"][::-1])
ax.set_xscale("log")
for i, v in enumerate(lc["count"][::-1]):
    ax.text(v * 1.05, i, f"{v:,}", va="center", fontsize=9)
ax.set_title(f"figW01 - every label occurrence ({TAG})")
save(fig, "figW01_label_counts_all.png")

fig, ax = plt.subplots(figsize=(8, 6))
top = lc.head(8)
other = lc["count"][8:].sum()
pie = pd.concat([top, pd.DataFrame(
    [{"label": "Other", "count": other}])]) if other else top
ax.pie(pie["count"], labels=pie["label"], autopct="%1.1f%%", startangle=90)
ax.set_title(f"figW02 - label share ({TAG})")
save(fig, "figW02_label_share.png")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, fi, f in zip(axes.ravel(), range(len(HIST_FEATS)), HIST_FEATS):
    for j, lab in enumerate(labels):
        ax.stairs(H[fi, j, :NB], BINS, label=lab, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f, fontsize=10)
    ax.tick_params(labelsize=8)
handles, lege = axes.ravel()[0].get_legend_handles_labels()
fig.legend(handles, lege, loc="center right", fontsize=9)
fig.suptitle(f"figW03 - full-data distributions per label "
             f"(log-log; zeros/inf in figW04)\n{TAG}")
save(fig, "figW03_histograms_all_labels.png", suptitle=True)

zrows = []
tc = {l: label_counts[l] for l in labels}
for i, f in enumerate(HIST_FEATS):
    for j, lab in enumerate(labels):
        z = H[i, j, I_ZERO] + H[i, j, I_UNDER]
        zrows.append({"feature": f, "label": lab,
                      "zero_pct": 100 * z / max(tc[lab], 1)})
zdf = pd.DataFrame(zrows)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=zdf, x="feature", y="zero_pct", hue="label", ax=ax)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
ax.set_ylabel("% rows with exact 0 (or <0.001)")
ax.set_title(f"figW04 - zero-value share per feature x label ({TAG})")
save(fig, "figW04_zero_share.png")

tdf = pd.DataFrame([{"t": t, "label": l, "count": n}
                    for (t, l), n in time_counts.items()])
fig, ax = plt.subplots(figsize=(12, 6))
if len(tdf):
    piv = tdf.pivot_table(index="t", columns="label", values="count",
                          aggfunc="sum").fillna(0).sort_index()
    piv.plot.area(ax=ax, alpha=0.7)
ax.set_xlabel("hour index (day*24+hour)")
ax.set_ylabel("flows")
ax.set_title(f"figW05 - every attack occurrence over time, stacked ({TAG})")
save(fig, "figW05_attacks_over_time_all.png")

pc = pd.DataFrame([{"port": p, "label": l, "count": n}
                   for (p, l), n in port_counts.items() if p != "INVALID"])
top_ports = pc.groupby("port")["count"].sum().nlargest(12).index.tolist()
pc = pc[pc["port"].isin(top_ports)]
if ben is not None and len(labels) > 6:
    pc["hue"] = pc["label"].apply(lambda x: "Benign" if x == ben else "Attack")
else:
    pc["hue"] = pc["label"]
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=pc, x="port", y="count", hue="hue", ax=ax)
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
ax.set_yscale("log")
ax.set_title(f"figW06 - top-12 ports x label "
             f"(INVALID ports: {range_bad['port_invalid']:,}) ({TAG})")
save(fig, "figW06_top_ports_all.png")

rg = pd.DataFrame([{"check": k, "fails": v} for k, v in range_bad.items()])
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=rg, x="check", y="fails", ax=ax)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
ax.set_title(f"figW07 - range violations over all rows ({TAG})")
save(fig, "figW07_range_violations_all.png")

print("DONE - 7 whole-data charts in figures/figW*.png")
