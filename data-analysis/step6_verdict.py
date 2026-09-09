"""
STEP 6 - VERDICT (assemble every report into one gate document)
Reads all reports/*.csv made by Steps 3-5 and writes reports/VERDICT.md:
  usable rows | drop/review columns | 32-feature status | class weights |
  go / no-go checklist for feature extraction.
Usage:
    python step6_verdict.py
"""
import os

import pandas as pd

os.makedirs("reports", exist_ok=True)

need = ["reports/quality_table_83cols.csv",
        "reports/label_distribution.csv",
        "reports/duplicates_summary.csv",
        "reports/line_audit.csv",
        "reports/invalid_ranges.csv"]
for f in need:
    if not os.path.exists(f):
        raise SystemExit(f"ERROR: {f} missing. Run Steps 3-4 first.")

FEATURES32 = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
    'Fwd Pkt Len Mean', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean',
    'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Fwd IAT Std', 'Bwd IAT Mean', 'Fwd Header Len',
    'Bwd Header Len', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Size Avg',
    'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
    'Fwd Seg Size Min', 'Down/Up Ratio', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt', 'Idle Mean', 'Active Mean']

q = pd.read_csv("reports/quality_table_83cols.csv")
dist = pd.read_csv("reports/label_distribution.csv")
dup = dict(zip(pd.read_csv("reports/duplicates_summary.csv")["metric"],
               pd.read_csv("reports/duplicates_summary.csv")["value"]))
line = dict(zip(pd.read_csv("reports/line_audit.csv")["reason"],
                pd.read_csv("reports/line_audit.csv")["count"]))

for c in ["missing_pct", "non_numeric", "inf", "neg", "finite_n"]:
    q[c] = pd.to_numeric(q[c], errors="coerce").fillna(0)
q["neg_adj"] = q["neg"]
win = q["column"].isin(["Init Fwd Win Byts", "Init Bwd Win Byts"])
q.loc[win, "neg_adj"] = (q.loc[win, "neg"]
                         - pd.to_numeric(q.loc[win, "neg1_sentinel"],
                                         errors="coerce").fillna(0))
q["constant"] = (q["min"] == q["max"]) & (q["finite_n"] > 0)
q["drop"] = (q["missing_pct"] > 30) | (q["column"] != "Label") & q["constant"]
q["review"] = (~q["drop"]) & ((q["inf"] > 0) | (q["neg_adj"] > 0)
                              | (q["non_numeric"] > 0))

parsed = int(dup.get("parsed_rows", 0))
ragged = int(float(dup.get("ragged_sparse_rows_dropped", 0)))
dups = int(float(dup.get("exact_duplicate_rows", 0)))
miss_lab = int(float(dup.get("missing_label_rows", 0)))
conf_rows = 0
n_conf_hashes = 0
if os.path.exists("reports/label_conflicts.csv"):
    try:
        cf = pd.read_csv("reports/label_conflicts.csv")
        if len(cf) and "occurrences" in cf:
            conf_rows = int(cf["occurrences"].sum())
            n_conf_hashes = len(cf)
    except pd.errors.EmptyDataError:
        pass
usable = parsed - dups - miss_lab - conf_rows

status32 = []
for f32 in FEATURES32:
    r = q[q["column"] == f32]
    if len(r) == 0:
        status32.append((f32, "-", "-", "-", "-", "MISSING-FROM-FILE"))
        continue
    r = r.iloc[0]
    v = ("DROP" if r["drop"] else
         "REVIEW" if r["review"] else "OK")
    status32.append((f32, f'{r["missing_pct"]:.2f}%', int(r["inf"]),
                     int(r["neg_adj"]), int(r["non_numeric"]), v))

ben = dist.loc[dist["label"].str.lower() == "benign", "count"]
benign_n = int(ben.iloc[0]) if len(ben) else 0

L = []
L.append("# DATA QUALITY VERDICT (gate to feature extraction)\n")
L.append("## Row accounting\n")
L.append(f"- Parsed rows (Step 3B): **{parsed:,}**")
L.append(f"- ok-shape rows (Step 3A): "
         f"{int(line.get('ok-shape', 0)):,}")
for k in ["short-row", "long-row", "terminal-junk", "empty-line",
          "mid-file-header"]:
    L.append(f"- {k}: {int(line.get(k, 0)):,}")
L.append(f"- Exact duplicate rows: {dups:,}")
L.append(f"- Missing-label rows: {miss_lab:,}")
L.append(f"- Label-conflict rows: {conf_rows:,}")
L.append(f"- **Usable rows ≈ {usable:,}**\n")
L.append("## Column decisions (all 83)\n")
drops = q[q["drop"]]["column"].tolist()
revs = q[q["review"]]["column"].tolist()
L.append(f"- DROP ({len(drops)}): "
         + (", ".join(drops) if drops else "none"))
L.append(f"- REVIEW ({len(revs)}): "
         + (", ".join(revs) if revs else "none"))
L.append(f"- OK ({int((~q['drop'] & ~q['review']).sum())}): rest\n")
L.append("## 32-feature status\n")
L.append("| feature | missing | inf | neg* | nonnum | verdict |")
L.append("|---|---|---|---|---|---|")
for row in status32:
    L.append("| " + " | ".join(str(x) for x in row) + " |")
L.append("\n*neg excludes the valid -1 sentinel in Init Win columns.\n")
L.append("## Labels + class weights\n")
for _, r in dist.iterrows():
    w = (round(benign_n / r["count"], 2)
         if r["count"] and benign_n and
         str(r["label"]).lower() != "benign" else "-")
    L.append(f"- {r['label']}: {int(r['count']):,} ({r['pct']}%) "
             f"scale_pos_weight={w} {r.get('rare_flag', '')}")
L.append("\n## Cleaning rules frozen for extraction\n")
L.append("- [ ] Quarantine rejects kept in rejects/ (never silently dropped)")
L.append("- [ ] Duplicates: drop keep-first; conflicts: quarantine")
L.append("- [ ] Missing Label rows: drop")
L.append("- [ ] inf -> cap to finite max + add had_inf flag (do in extraction)")
L.append("- [ ] -1 in Init Win kept as valid category, not missing")
L.append("- [ ] Day/Hour/Minute/Second + Dst Port/Protocol: analysis only, "
         "train behaviour-only + with-port variants and report both\n")
L.append("## Gate\n")
L.append("Proceed to feature extraction ONLY if: usable rows counted, "
         "every DROP/REVIEW above has an owner decision, and no empty "
         "section remains in this file.")

open("reports/VERDICT.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print("\nSaved reports/VERDICT.md — read it, then start feature extraction.")
