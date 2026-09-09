"""
STEP 3A - RAW LINE AUDIT (exact row-shape + junk classification with line numbers)
Streams the file once as text. Classifies EVERY line:
  true-header | ok-shape | mid-file-header | terminal-junk | empty-line |
  short-row | long-row
Usage:
    python step3a_lineaudit.py [merged_data.csv]
Outputs:
    rejects/rejected_lines.csv   (line_no, reason, fields, preview — bad rows only)
    reports/line_audit.csv       (reason, count)
    reports/field_histogram.csv  (field_count, rows)
Note: field counting assumes no quoted commas inside fields (verified in Step 2).
"""
import csv
import os
import sys
from collections import Counter
from tqdm import tqdm

FILE = sys.argv[1] if len(sys.argv) > 1 else "merged_data.csv"
os.makedirs("rejects", exist_ok=True)
os.makedirs("reports", exist_ok=True)

JUNK_STARTS = ("wc ", "wc\t", "tail ", "tail\t", "---", "===")
REJECT_CAP = 100000  # cap quarantine file size; counts still cover all rows


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
NCOLS = len(cols)
print(f"Header at line {hidx + 1}, expecting {NCOLS} fields per row.")

total = 0
with open(FILE, "rb") as f:
    for blk in iter(lambda: f.read(1024 * 1024), b""):
        total += blk.count(b"\n")

reasons = Counter()
field_hist = Counter()
examples = {}
rej_written = 0

with open(FILE, "r", encoding="utf-8", errors="replace") as f, \
        open("rejects/rejected_lines.csv", "w", newline="",
             encoding="utf-8") as r:
    w = csv.writer(r)
    w.writerow(["line_no", "reason", "fields", "preview_200chars"])
    for idx, line in enumerate(tqdm(f, total=total, unit=" lines",
                                    desc="Auditing lines")):
        if idx == hidx:
            reasons["true-header"] += 1
            continue
        s = line.strip()
        if s == "":
            reasons["empty-line"] += 1
            continue
        if s.startswith("Dst Port,"):
            reasons["mid-file-header"] += 1
            if rej_written < REJECT_CAP:
                w.writerow([idx + 1, "mid-file-header", s.count(",") + 1,
                            s[:200]])
                rej_written += 1
            continue
        if s.startswith(JUNK_STARTS) or "merged_data.csv" in s[:80]:
            reasons["terminal-junk"] += 1
            examples.setdefault("terminal-junk", s[:200])
            if rej_written < REJECT_CAP:
                w.writerow([idx + 1, "terminal-junk", s.count(",") + 1,
                            s[:200]])
                rej_written += 1
            continue
        nf = s.count(",") + 1
        field_hist[nf] += 1
        if nf == NCOLS:
            reasons["ok-shape"] += 1
        elif nf < NCOLS:
            reasons["short-row"] += 1
            examples.setdefault("short-row", s[:200])
            if rej_written < REJECT_CAP:
                w.writerow([idx + 1, f"short-row({nf}<{NCOLS})", nf, s[:200]])
                rej_written += 1
        else:
            reasons["long-row"] += 1
            examples.setdefault("long-row", s[:200])
            if rej_written < REJECT_CAP:
                w.writerow([idx + 1, f"long-row({nf}>{NCOLS})", nf, s[:200]])
                rej_written += 1

with open("reports/line_audit.csv", "w", newline="", encoding="utf-8") as o:
    w = csv.writer(o)
    w.writerow(["reason", "count"])
    for k, v in reasons.most_common():
        w.writerow([k, v])

with open("reports/field_histogram.csv", "w", newline="",
          encoding="utf-8") as o:
    w = csv.writer(o)
    w.writerow(["field_count", "rows"])
    for k in sorted(field_hist):
        w.writerow([k, field_hist[k]])

print("\n==== LINE AUDIT ====")
for k, v in reasons.most_common():
    print(f"  {k:16s} {v:,}")
print(f"Quarantined rows written: {rej_written:,} "
      f"(cap {REJECT_CAP:,}; full counts above cover everything)")
for k, v in examples.items():
    print(f"  example {k}: {v}")
print("Saved reports/line_audit.csv, reports/field_histogram.csv, "
      "rejects/rejected_lines.csv")
print("NEXT: python step3b_profile.py")
