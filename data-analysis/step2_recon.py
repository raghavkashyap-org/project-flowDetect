"""
STEP 2 - FILE RECON (fast: head/middle/tail + line count + header hunt)
Usage:
    python step2_recon.py [merged_data.csv]
Output:
    reports/file_recon.md   (read this before Step 3)
"""
import os
import sys
from collections import deque

FILE = sys.argv[1] if len(sys.argv) > 1 else "merged_data.csv"
os.makedirs("reports", exist_ok=True)


def find_header(path, max_scan=200):
    """Return (0-based line index, column list, header text)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                break
            if line.startswith("Dst Port,"):
                cols = [c.strip() for c in line.strip().split(",")]
                return i, cols, line.strip()
    raise SystemExit(
        f"ERROR: header starting with 'Dst Port,' not found in first "
        f"{max_scan} lines of {FILE}. Check you pointed at the right file.")


print(f"File: {FILE}")
size = os.path.getsize(FILE)
print(f"Size: {size / 1e9:.2f} GB")

# 1) fast total line count (binary, ~20-40 s on 5.8GB)
total = 0
with open(FILE, "rb") as f:
    for blk in iter(lambda: f.read(1024 * 1024), b""):
        total += blk.count(b"\n")
print(f"Total lines (newline count): {total:,}")

# 2) line-ending + encoding probe
with open(FILE, "rb") as f:
    probe = f.read(500_000)
CRLF = b"\r\n"
LF = b"\n"
print(f"First 500KB: CRLF={probe.count(CRLF):,}  LF={probe.count(LF):,}")
try:
    with open(FILE, "r", encoding="utf-8", errors="strict") as f:
        f.read(1_000_000)
    enc_note = "UTF-8 strict OK on first 1MB"
except UnicodeDecodeError as e:
    enc_note = f"NOT strict UTF-8 ({e}); scripts use errors='replace'"
print("Encoding:", enc_note)

# 3) header hunt
hidx, cols, htext = find_header(FILE)
print(f"Header at line {hidx + 1} (1-based). Columns: {len(cols)}")
print(f"  first cols: {cols[:5]}")
print(f"  last cols:  {cols[-5:]}")

# 4) count mid-file header repeats (full scan, cheap compare)
mid = 0
with open(FILE, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("Dst Port,"):
            mid += 1
print(f"'Dst Port,' lines: {mid} = 1 true header + {mid - 1} mid-file repeats")

# 5) head / middle / tail peek
with open(FILE, "r", encoding="utf-8", errors="replace") as f:
    head = [f.readline().rstrip("\n") for _ in range(min(8, hidx + 4))]
with open(FILE, "r", encoding="utf-8", errors="replace") as f:
    tail = list(deque(f, maxlen=5))
with open(FILE, "rb") as f:
    f.seek(max(0, size // 2))
    f.readline()  # sync to next full line
    middle = [ln.decode("utf-8", errors="replace").rstrip("\n")
              for ln in (f.readline() for _ in range(3))]

# 6) delimiter sanity on first data lines
print("Comma counts after header (expect 82 for 83 cols):")
with open(FILE, "r", encoding="utf-8", errors="replace") as f:
    for _ in range(hidx + 1):
        f.readline()
    for k in range(5):
        ln = f.readline()
        if not ln:
            break
        print(f"  data line {k + 1}: commas={ln.count(',')} chars={len(ln.strip())}")

with open("reports/file_recon.md", "w", encoding="utf-8") as o:
    o.write(f"# File recon — {FILE}\n\n")
    o.write(f"- Size: {size / 1e9:.2f} GB\n")
    o.write(f"- Total lines: {total:,}\n")
    o.write(f"- Encoding: {enc_note}\n")
    o.write(f"- Header at 1-based line {hidx + 1}, columns = {len(cols)}\n")
    o.write(f"- Mid-file header repeats: {mid - 1} (must be skipped in analysis)\n")
    o.write(f"- Expected data rows ≈ {total - (hidx + 1) - (mid - 1):,} "
            f"(minus junk/short/long rows found in Step 3A)\n\n")
    o.write("## Head\n```\n" + "\n".join(head) + "\n```\n\n")
    o.write("## Middle (50% byte offset)\n```\n" + "\n".join(middle) + "\n```\n\n")
    o.write("## Tail\n```\n" + "".join(tail) + "```\n")

print("Saved reports/file_recon.md")
print("NEXT: python step3a_lineaudit.py")
