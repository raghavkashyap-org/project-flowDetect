# 5.8GB / 16M-Row Analysis — Runbook (12GB RAM, i5)

**Setup:** Copy all `step*.py` files into the SAME folder as `merged_data.csv`.
Install once: `pip install -r requirements.txt` (Python 3.10 or 3.11, 64-bit).

On Windows use `python`, on Mac/Linux use `python3` in every command below.

## Run order (do not skip or reorder)

| Step | Command | Time on 5.8GB | Produces |
|---|---|---|---|
| 1. Setup | see commands below | 2 min | folders |
| 2. Recon | `python step2_recon.py` | ~1–2 min | `reports/file_recon.md` |
| 3A. Line audit | `python step3a_lineaudit.py` | ~3–8 min | `reports/line_audit.csv`, `rejects/rejected_lines.csv` |
| 3B. Profiling | `python step3b_profile.py` | ~15–30 min | `reports/quality_table_83cols.csv`, `label_raw_counts.csv`, `duplicates_summary.csv`, `logs/` |
| 4. Labels | `python step4_labels.py` | <1 min | `reports/label_distribution.csv` ← attack types answer |
| 5A. Sample | `python step5a_sample.py` | ~5–10 min | `data/sample_100k.parquet` |
| 5B. Visuals | `python step5b_visuals.py` | ~2–5 min | `figures/*.png` (11 charts) |
| 6. Verdict | `python step6_verdict.py` | <1 min | `reports/VERDICT.md` ← gate to feature extraction |

Optional flags: `python step3b_profile.py --chunksize 100000` (if RAM >9GB),
`--no-global-dedup` (skip global duplicate tracking on tight RAM),
`--file other.csv` (any script, to analyse a different file, e.g. the demo).

## Step 1 commands

Windows (CMD or PowerShell):
```
mkdir reports figures rejects logs data
python --version
dir merged_data.csv
```

Mac / Linux:
```
mkdir -p reports figures rejects logs data
python3 --version
ls -lh merged_data.csv
```

Verify install (all systems):
```
python -c "import pandas, numpy, matplotlib, seaborn, tqdm, psutil, pyarrow, openpyxl, scipy; print('all deps OK')"
```

## Verify each step before moving on

- Step 2: `reports/file_recon.md` shows Columns = 83 and header line found.
- Step 3A: console shows `ok-shape` ≈ 16M; every other category is small and saved with line numbers.
- Step 3B: RAM column in console stays under ~9GB; ends with "STEP 3B DONE".
- Step 4: console prints the attack-types table with counts and %.
- Step 5A: quota vs taken table — every label filled.
- Step 5B: 11 PNG files in `figures/`, every title says the sample size.
- Step 6: `reports/VERDICT.md` filled with numbers, no empty sections.

## Troubleshooting

| Symptom | Fix |
|---|---|
| RAM crosses 9–10GB in Step 3B | Stop, rerun with `--chunksize 100000`; script also auto-disables global dedup past 10GB and records it |
| `UnicodeDecodeError` | Scripts already use `errors='replace'`; if a custom read fails, add the same flag |
| Step 3B slow (>45 min) | Normal on HDD; leave it running, progress bar shows ETA. Close Chrome/Excel to free RAM |
| `ModuleNotFoundError` | Rerun `pip install -r requirements.txt` in the same Python you run scripts with |
| Header not found error | Open the first 10 lines of the CSV and check the file is the right one / not renamed |
| `dup_hashes.npy` missing warning in 5A | Means dedup was disabled in 3B; conflict check is skipped and noted in the verdict |

## After Step 6

Read `reports/VERDICT.md`. Only when every gate item is answered do you proceed to
feature extraction (your 32 Tier-1/2 features + cleaning rules from the verdict).
