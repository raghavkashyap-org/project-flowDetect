"""Proves --dedup-key full vs features behaves as documented.

Works whether this file sits in the project root OR in a test/ subfolder:
it locates 1_filter_clean.py next to itself, or one directory up.

Builds a small CSV where:
  * 5 rows are TRUE duplicates (identical in ALL measurement columns)
  * 20 rows are identical in the 32 kept features but DIFFER in a non-kept
    measurement column (the FTP-BruteForce situation)
Then checks:
  full      -> drops 4 (keeps 1 of the 5 true dupes); keeps all 20 near-twins
  features  -> additionally collapses the 20 near-twins to 1
"""
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def find_script(name="1_filter_clean.py"):
    """Look for the pipeline script beside this file, then one level up."""
    for cand in (os.path.join(HERE, name),
                 os.path.join(os.path.dirname(HERE), name)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"{name} not found next to {HERE} or one level up. "
        "Keep this test in the same folder as the pipeline scripts.")


SCRIPT = find_script()
# common.py lives beside the script; make it importable
sys.path.insert(0, os.path.dirname(SCRIPT))
from common import FEATURES, LABEL_COL  # noqa: E402


def build(path):
    nonkept = ["Tot Bwd Pkts", "Subflow Fwd Pkts", "Total Length of Bwd Packets"]
    cols = FEATURES + [LABEL_COL] + nonkept
    rows = []

    # 5 true duplicates (all measurement cols identical)
    base = {c: 10.0 for c in FEATURES}
    base.update({c: 5.0 for c in nonkept})
    for _ in range(5):
        r = dict(base)
        r[LABEL_COL] = "FTP-BruteForce"
        rows.append(r)

    # 20 near-twins: same 32 features, different non-kept measurement col
    for k in range(20):
        r = {c: 100.0 for c in FEATURES}
        r["Tot Bwd Pkts"] = float(k)
        r["Subflow Fwd Pkts"] = float(k)
        r["Total Length of Bwd Packets"] = float(k)
        r[LABEL_COL] = "FTP-BruteForce"
        rows.append(r)

    pd.DataFrame(rows)[cols].to_csv(path, index=False, lineterminator="\n")


def run(mode, csv, outdir):
    subprocess.run([sys.executable, SCRIPT,
                    "--input", csv, "--outdir", outdir,
                    "--dedup-key", mode],
                   check=True, capture_output=True, text=True)
    with open(os.path.join(outdir, "filter_report.json")) as f:
        return json.load(f)


def main():
    print(f"using pipeline script: {SCRIPT}")
    csv = os.path.join(HERE, "_t.csv")
    build(csv)
    ok = True
    for mode, exp_kept, exp_dup in (("full", 21, 4), ("features", 2, 23)):
        outdir = os.path.join(HERE, f"_out_{mode}")
        rep = run(mode, csv, outdir)
        kept = rep["clean_rows_written"]
        dup = rep["dropped"].get("duplicate", 0)
        good = (kept == exp_kept and dup == exp_dup)
        ok &= good
        print(f"  [{mode:>8}] kept={kept:<4} duplicate_dropped={dup:<4} "
              f"expected kept={exp_kept} dup={exp_dup}  "
              f"{'PASS' if good else 'FAIL'}")

    for p in ("_t.csv", "_out_full", "_out_features"):
        fp = os.path.join(HERE, p)
        if os.path.isdir(fp):
            shutil.rmtree(fp)
        elif os.path.exists(fp):
            os.remove(fp)
    print("DEDUP-KEY TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
