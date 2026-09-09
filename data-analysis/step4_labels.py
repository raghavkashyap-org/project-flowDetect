"""
STEP 4 - LABEL NORMALIZATION + ATTACK TYPES (fast: works only on small tables)
Reads Step 3B outputs, strips whitespace/case noise from labels, and answers:
  "how many types of attacks are there?" with exact counts, %, imbalance ratios.
Usage:
    python step4_labels.py
Inputs:
    reports/label_raw_counts.csv, reports/port_label_top.csv, reports/time_label.csv
Outputs:
    reports/label_mapping.csv        (raw_label -> canonical)
    reports/label_distribution.csv   (label, count, pct, imbalance_vs_benign, rare_flag)
    reports/port_label_canonical.csv (port x canonical label)
    reports/attacks_per_hour.csv     (day x hour x canonical label)
"""
import os

import pandas as pd

os.makedirs("reports", exist_ok=True)

for f in ["reports/label_raw_counts.csv", "reports/port_label_top.csv",
          "reports/time_label.csv"]:
    if not os.path.exists(f):
        raise SystemExit(f"ERROR: {f} missing. Run Step 3B first.")

raw = pd.read_csv("reports/label_raw_counts.csv")
raw["raw_label"] = raw["raw_label"].astype(str)
raw["stripped"] = raw["raw_label"].str.strip()
raw["key"] = raw["stripped"].str.lower()

# canonical = stripped form (preserves real names like 'FTP-BruteForce');
# rows that differ ONLY by space/case collapse to one canonical label.
mapping = raw[["raw_label", "stripped"]].drop_duplicates()
mapping.columns = ["raw_label", "canonical"]
mapping.to_csv("reports/label_mapping.csv", index=False)

print("Labels differing only by space/case (auto-merged):")
merged_any = False
for key, g in raw.groupby("key"):
    forms = sorted(g["stripped"].unique().tolist())
    if len(forms) > 1:
        merged_any = True
        print(f"  {forms} -> '{forms[0]}'")
if not merged_any:
    print("  none — all raw labels already clean")

dist = raw.groupby("stripped", as_index=False)["count"].sum()
dist.columns = ["label", "count"]
dist = dist.sort_values("count", ascending=False).reset_index(drop=True)
total = int(dist["count"].sum())
dist["pct"] = (100 * dist["count"] / total).round(4)
ben = dist.loc[dist["label"].str.lower() == "benign", "count"]
benign_n = int(ben.iloc[0]) if len(ben) else 0
dist["imbalance_vs_benign"] = dist["count"].apply(
    lambda c: round(benign_n / c, 2) if c and benign_n else "")
dist["rare_flag"] = dist["count"].apply(
    lambda c: "RARE(<5000)" if c < 5000 else "")
dist.to_csv("reports/label_distribution.csv", index=False)

attacks = dist[dist["label"].str.lower() != "benign"]
print(f"\n==== ATTACK TYPES: {len(attacks)} (+ Benign) "
      f"over {total:,} rows ====")
print(dist.to_string(index=False))
print(f"\nscale_pos_weight hint (binary Benign vs all-attacks): "
      f"{round(benign_n / max(int(attacks['count'].sum()), 1), 2)}")


def disp_num(s):
    """'21.0'->'21' for display; leaves real strings alone."""
    s = s.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


# port x canonical label
pl = pd.read_csv("reports/port_label_top.csv")
pl["raw_label"] = pl["raw_label"].astype(str)
pl = pl.merge(mapping, on="raw_label", how="left")
pl["canonical"] = pl["canonical"].fillna(pl["raw_label"].str.strip())
pl["port"] = disp_num(pl["port"])
out = pl.groupby(["port", "canonical"], as_index=False)["count"].sum()
out = out.sort_values("count", ascending=False)
out.to_csv("reports/port_label_canonical.csv", index=False)
print("\nTop 10 port x label:")
print(out.head(10).to_string(index=False))

# day x hour x canonical label
tl = pd.read_csv("reports/time_label.csv")
tl["raw_label"] = tl["raw_label"].astype(str)
tl = tl.merge(mapping, on="raw_label", how="left")
tl["canonical"] = tl["canonical"].fillna(tl["raw_label"].str.strip())
tl["day"] = disp_num(tl["day"])
tl["hour"] = disp_num(tl["hour"])
out2 = tl.groupby(["day", "hour", "canonical"],
                  as_index=False)["count"].sum()
out2 = out2.sort_values(["day", "hour", "canonical"])
out2.to_csv("reports/attacks_per_hour.csv", index=False)
print(f"\nSaved reports/label_mapping.csv, label_distribution.csv, "
      f"port_label_canonical.csv, attacks_per_hour.csv")
print("NEXT: python step5a_sample.py")
