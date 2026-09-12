"""
Feature ablation -- how much of the accuracy survives when the tool
fingerprints are removed?

Your audit prescribed exactly this ("report accuracy with vs without
InitWin", section 9.3 / fig10) and it is the single most important number for
deciding whether the model has learned attack BEHAVIOUR or tool DEFAULTS.

Trains the same model on nested feature sets and reports the delta:

  all32            every Tier-1/2 feature
  no_initwin       minus the two Init*Win columns (the #1/#4 features)
  behavioural      minus the whole tool-fingerprint group
  handshake_timing only flags + IAT/active/idle (slow-attack signal)

Usage:
    python3 ablate.py --splits data_out/splits --outdir reports/ablation
    python3 ablate.py --splits data_out/splits --max-train-rows 1000000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from common import FEATURES, HGB_PARAMS
from train_model import (evaluate, load_split, plot_cm, stratified_subsample)

# The columns that are wholesale tool defaults rather than traffic behaviour.
# Grouped from the permutation-importance ranking and the audit's fig10 note
# ("FTP a single spike at 26883 in all flows; HOIC ~33k; tool-default
#  fingerprints are real and strong - but brittle").
TOOL_GROUP = [
    "Init Fwd Win Byts",     # TCP window default of the attack tool
    "Init Bwd Win Byts",     # server-side window / -1 sentinel
    "Fwd Seg Size Min",      # MSS default of the sending stack
    "Fwd Header Len",        # constant header size for a given tool
    "Pkt Size Avg",          # mechanically determined by the above
]

HANDSHAKE_TIMING = [
    "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt",
    "Fwd PSH Flags",
    "Flow IAT Mean", "Flow IAT Max", "Flow IAT Std",
    "Fwd IAT Std", "Bwd IAT Mean", "Bwd IAT Tot",
    "Idle Mean", "Active Mean", "Flow Duration",
]


def benign_fpr(r: dict) -> tuple[float, int, int]:
    """(false-positive rate on Benign, #Benign predicted as attack, #Benign rows).

    For an IDS this is the number that matters operationally: every one of
    these is an alert a human has to triage.
    """
    cm = np.array(r["confusion_matrix"])
    classes = list(r["per_class"].keys())
    if "Benign" not in classes:
        return (float("nan"), 0, 0)
    bi = classes.index("Benign")
    n_ben = int(cm[bi].sum())
    fp = n_ben - int(cm[bi, bi])
    return (fp / n_ben if n_ben else float("nan"), fp, n_ben)


def variants() -> dict[str, list[str]]:
    v = {
        "all32": list(FEATURES),
        "no_initwin": [f for f in FEATURES if not f.startswith("Init ")],
        "behavioural": [f for f in FEATURES if f not in TOOL_GROUP],
        "handshake_timing": list(HANDSHAKE_TIMING),
    }
    # nested check: every variant must be a subset of all32
    for k, cols in v.items():
        assert set(cols) <= set(FEATURES), f"{k} has unknown columns"
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data_out/splits")
    ap.add_argument("--outdir", default="reports/ablation")
    ap.add_argument("--max-train-rows", type=int, default=2_000_000,
                    help="0 = all rows")
    ap.add_argument("--max-val-rows", type=int, default=400_000,
                    help="0 = all rows (slower)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    from sklearn.ensemble import HistGradientBoostingClassifier

    Xtr, ytr = load_split(os.path.join(args.splits, "train.parquet"))
    Xva, yva = load_split(os.path.join(args.splits, "val.parquet"))
    Xtr, ytr = stratified_subsample(Xtr, ytr, args.max_train_rows, args.seed)
    Xva, yva = stratified_subsample(Xva, yva, args.max_val_rows, args.seed)
    classes = np.array(sorted(set(ytr) | set(yva)))
    idx = {f: i for i, f in enumerate(FEATURES)}
    print(f"train={len(ytr):,}  val={len(yva):,}  classes={len(classes)}")

    results = {}
    print(f"\n{'variant':<20}{'#feat':>7}{'bal.acc':>11}{'macroF1':>10}"
          f"{'worst-class F1':>17}")
    for name, cols in variants().items():
        ci = [idx[c] for c in cols]
        clf = HistGradientBoostingClassifier(
            max_iter=args.n_estimators, class_weight="balanced",
            random_state=args.seed, **HGB_PARAMS)
        clf.fit(Xtr[:, ci], ytr)
        r = evaluate(clf, Xva[:, ci], yva, classes, f"VAL [{name}]")
        worst = min(r["per_class"].items(), key=lambda kv: kv[1]["f1"])
        print(f"{name:<20}{len(cols):>7}{r['balanced_accuracy']:>11.6f}"
              f"{r['macro_f1']:>10.6f}"
              f"   {worst[0]} {worst[1]['f1']:.4f}")
        results[name] = {
            "n_features": len(cols), "features": cols,
            "balanced_accuracy": r["balanced_accuracy"],
            "macro_f1": r["macro_f1"],
            "accuracy": r["accuracy"],
            "per_class": r["per_class"],
            "confusion_matrix": r["confusion_matrix"],
        }
        plot_cm(np.array(r["confusion_matrix"]), r["classes"],
                os.path.join(args.outdir, f"cm_{name}.png"),
                f"VAL — {name} ({len(cols)} features)")

    base = results["all32"]["balanced_accuracy"]
    base_macro = results["all32"]["macro_f1"]
    print("\n" + "=" * 74)
    print("DELTA vs all32  (how much accuracy the tool fingerprints were worth)")
    print("=" * 74)
    for name, r in results.items():
        db = r["balanced_accuracy"] - base
        dm = r["macro_f1"] - base_macro
        print(f"  {name:<20} bal {r['balanced_accuracy']:.6f} "
              f"({'+' if db >= 0 else ''}{db:.6f})   "
              f"macroF1 {r['macro_f1']:.6f} "
              f"({'+' if dm >= 0 else ''}{dm:.6f})")

    # ---- security cost of dropping the fingerprints ----------------------
    print("\n" + "=" * 74)
    print("SECURITY COST of the behavioural variant")
    print("=" * 74)
    for name in ("all32", "behavioural", "handshake_timing"):
        r = results[name]
        fpr, fp, n_ben = benign_fpr(r)
        precs = {c: v["precision"] for c, v in r["per_class"].items()}
        worst = min((c for c in precs if c != "Benign"),
                    key=lambda c: precs[c])
        print(f"  {name:<20} benign FPR {fpr:7.4%}  ({fp:,}/{n_ben:,})   "
              f"worst attack precision: {worst} {precs[worst]:.4f}")

    b = results["behavioural"]
    macro_drop = base_macro - b["macro_f1"]
    fpr_b, fp_b, n_ben_b = benign_fpr(b)
    prec_b = {c: v["precision"] for c, v in b["per_class"].items()
              if c != "Benign"}
    worst_c = min(prec_b, key=lambda c: prec_b[c])
    print("\n" + "-" * 74)
    print(f"  macro F1 drop          : {macro_drop:+.6f}")
    print(f"  benign false positives : {fp_b:,} of {n_ben_b:,} "
          f"({fpr_b:.4%}) — these are the alerts an analyst must triage")
    print(f"  weakest attack class   : {worst_c} precision "
          f"{prec_b[worst_c]:.4f} "
          f"(meaning {1-prec_b[worst_c]:.1%} of its alerts are wrong)")
    if macro_drop > 0.05 or fpr_b > 0.01 or prec_b[worst_c] < 0.85:
        print("\n  VERDICT: SIGNIFICANT degradation without the tool fingerprints.")
        print("  A meaningful share of the headline score was tool identity. Report")
        print("  the behavioural variant as primary, and check whether the classes")
        print("  that collapse are separable on behaviour at all -- if not, group")
        print("  them into a family class (see the audit's hierarchy).")
    else:
        print("\n  VERDICT: the signal largely survives without the tool group; the")
        print("  headline number is broadly defensible.")
    print("-" * 74)

    with open(os.path.join(args.outdir, "ablation.json"), "w") as fh:
        json.dump({"tool_group": TOOL_GROUP, "results": results}, fh, indent=2)
    print(f"\nelapsed {time.time()-t0:.1f}s -> {args.outdir}/ablation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
