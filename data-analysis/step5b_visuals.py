"""
STEP 5B - CHARTS (everything from the 100k sample + full-data report tables)
11 PNG figures, static + lightweight for i5 (matplotlib Agg + seaborn).
Usage:
    python step5b_visuals.py
Inputs:
    data/sample_100k.parquet, reports/quality_table_83cols.csv,
    reports/attacks_per_hour.csv
Outputs:
    figures/fig00_...fig10_*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
os.makedirs("figures", exist_ok=True)

for f in ["data/sample_100k.parquet", "reports/quality_table_83cols.csv",
          "reports/attacks_per_hour.csv"]:
    if not os.path.exists(f):
        raise SystemExit(f"ERROR: {f} missing. Run Steps 3B, 4, 5A first.")

FEATURES32 = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
    'Fwd Pkt Len Mean', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean',
    'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Fwd IAT Std', 'Bwd IAT Mean', 'Fwd Header Len',
    'Bwd Header Len', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Size Avg',
    'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
    'Fwd Seg Size Min', 'Down/Up Ratio', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt', 'Idle Mean', 'Active Mean']

df = pd.read_parquet("data/sample_100k.parquet")
df["Label"] = df["Label"].astype(str)
N = len(df)
print(f"Sample rows: {N:,}, labels: {sorted(df['Label'].unique())}")
num = df.replace([np.inf, -np.inf], np.nan)
TAG = f"n={N:,} stratified sample"


def save(fig, name, suptitle=False):
    if suptitle:
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    else:
        fig.tight_layout()
    fig.savefig(f"figures/{name}", dpi=120)
    plt.close(fig)
    print(" ", name)


# fig00 — data-quality scorecard for the 32 features (FULL-data percentages)
q = pd.read_csv("reports/quality_table_83cols.csv")
q32 = q[q["column"].isin(FEATURES32)].copy()
for c in ["missing_pct", "inf", "neg", "non_numeric", "parsed_rows"]:
    q32[c] = pd.to_numeric(q32[c], errors="coerce").fillna(0)
tot = q32["parsed_rows"].max()
q32["inf_pct"] = 100 * q32["inf"] / tot
q32["neg_pct"] = 100 * q32["neg"] / tot
q32["bad_pct"] = 100 * q32["non_numeric"] / tot
q32["ok_pct"] = (100 - q32["missing_pct"] - q32["inf_pct"]
                 - q32["neg_pct"] - q32["bad_pct"]).clip(lower=0)
q32 = q32.sort_values("ok_pct")
fig, ax = plt.subplots(figsize=(10, 9))
left = np.zeros(len(q32))
for col, lab in [("ok_pct", "OK"), ("missing_pct", "missing"),
                 ("inf_pct", "inf"), ("neg_pct", "negative"),
                 ("bad_pct", "non-numeric")]:
    ax.barh(q32["column"], q32[col], left=left, label=lab)
    left = left + q32[col].values
ax.set_xlabel("% of all rows (full data; overlaps ignored)")
ax.set_title(f"fig00 — 32-feature quality scorecard (full data)")
ax.legend(loc="lower right")
save(fig, "fig00_quality_scorecard_32.png")

# fig01 — label counts, log scale
order = df["Label"].value_counts().index.tolist()
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(y="Label", data=df, order=order, ax=ax)
ax.set_xscale("log")
for p in ax.patches:
    ax.text(p.get_width() * 1.05, p.get_y() + p.get_height() / 2,
            f"{int(p.get_width()):,}", va="center", fontsize=9)
ax.set_title(f"fig01 — label counts, log x-axis ({TAG})")
save(fig, "fig01_label_counts_log.png")

# fig02 — Duration vs Pkts/s, log-log, THE attack-corner chart
d2 = num[(num["Flow Duration"] > 0) & (num["Flow Pkts/s"] > 0)]
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=d2, x="Flow Duration", y="Flow Pkts/s", hue="Label",
                alpha=0.35, s=12, ax=ax, rasterized=True)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title(f"fig02 — short+fast = attack corner ({TAG}, positive only)")
save(fig, "fig02_duration_vs_pktss_loglog.png")

# fig03 — forward vs backward bytes (log1p keeps the (0,0) cluster visible)
fig, ax = plt.subplots(figsize=(10, 6))
fx = np.log1p(num["TotLen Fwd Pkts"].fillna(0).clip(lower=0))
bx = np.log1p(num["TotLen Bwd Pkts"].fillna(0).clip(lower=0))
sns.scatterplot(x=fx, y=bx, hue=df["Label"], alpha=0.35, s=12, ax=ax,
                rasterized=True)
ax.set_xlabel("log1p(TotLen Fwd Pkts)")
ax.set_ylabel("log1p(TotLen Bwd Pkts)")
ax.set_title(f"fig03 — fwd vs bwd bytes; (0,0) = header-only flows ({TAG})")
save(fig, "fig03_fwd_vs_bwd_bytes.png")

# fig04 — 2x2 distribution grid, log x, hue=label
feats4 = ["Flow Duration", "Flow Pkts/s", "Flow IAT Mean", "Pkt Len Mean"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), feats4):
    d = num[num[col] > 0]
    sns.histplot(data=d, x=col, hue="Label", element="step",
                 common_norm=False, log_scale=(True, False), ax=ax)
    ax.set_title(col)
fig.suptitle(f"fig04 — Tier-1 distributions by label\n{TAG}, positive only")
save(fig, "fig04_tier1_distributions.png", suptitle=True)

# fig05 — 32-feature correlation heatmap
c32 = num[[c for c in FEATURES32 if c in num]].apply(pd.to_numeric,
                                                    errors="coerce")
corr = c32.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title(f"fig05 — 32-feature correlation, lower triangle ({TAG})")
save(fig, "fig05_correlation_32.png")

# fig06 — worst-20 missing (FULL data)
qtop = q.copy()
qtop["missing_pct"] = pd.to_numeric(qtop["missing_pct"], errors="coerce")
qtop = qtop.sort_values("missing_pct", ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=qtop, y="column", x="missing_pct", ax=ax)
ax.set_xlabel("missing % (full data)")
ax.set_title("fig06 — top-20 columns by missing % (full data)")
save(fig, "fig06_missing_top20.png")

# fig07 — missingness map on 5k sample rows x 32 cols
m5 = df[[c for c in FEATURES32 if c in df]].isna()
m5 = m5.sample(min(5000, len(m5)), random_state=42)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(m5, cbar=True, yticklabels=False, ax=ax)
ax.set_xlabel("32 features")
ax.set_title(f"fig07 — missing map, dark cell = missing "
             f"(n={len(m5):,} sample rows)")
save(fig, "fig07_missing_heatmap.png")

# fig08 — attacks over time (FULL data; attacks only)
t = pd.read_csv("reports/attacks_per_hour.csv")
t["day_n"] = pd.to_numeric(t["day"], errors="coerce")
t["hour_n"] = pd.to_numeric(t["hour"], errors="coerce")
t = t.dropna(subset=["day_n", "hour_n"])
t["t"] = t["day_n"] * 24 + t["hour_n"]
t = t[t["canonical"].str.lower() != "benign"]
fig, ax = plt.subplots(figsize=(12, 6))
if len(t):
    sns.lineplot(data=t.sort_values("t"), x="t", y="count",
                 hue="canonical", marker="o", ax=ax)
ax.set_xlabel("hour index (day*24+hour)")
ax.set_title("fig08 — attack bursts over time, attacks only (full data)")
save(fig, "fig08_attacks_over_time.png")

# fig09 — flag rates per label (sample means)
flags = ["SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt"]
fm = num.groupby("Label")[flags].mean(numeric_only=True).reset_index()
fm = fm.melt("Label", var_name="flag", value_name="mean_rate")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=fm, x="Label", y="mean_rate", hue="flag", ax=ax)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
ax.set_title(f"fig09 — mean TCP-flag rates per label ({TAG})")
save(fig, "fig09_flag_rates.png")

# fig10 — Init Fwd Win by label (tool fingerprints)
w = num[(num["Init Fwd Win Byts"] >= 0)
        & (num["Init Fwd Win Byts"] <= 70000)]
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=w, y="Label", x="Init Fwd Win Byts", ax=ax)
ax.set_title(f"fig10 — forward window size by label, spike = tool "
             f"fingerprint ({TAG})")
save(fig, "fig10_initwin_by_label.png")

print("NEXT: python step6_verdict.py")
