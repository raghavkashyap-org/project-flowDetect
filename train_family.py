"""
Family-level training + flood-family separability proof.

Two things this script establishes:

1. FAMILY MODEL -- retrains the task with the 6 classes grouped into families,
   because the ablation showed the three flood classes (HOIC / LOIC-HTTP /
   Hulk) collapse into each other once per-tool packet templates are removed.
   A family label ("Flood") is what the behavioural features can actually
   support.

2. PAIRWISE PROOF -- trains one binary classifier per pair among the flood
   classes, on all32 and on behavioural features. If two flood classes are
   only separable WITH the tool templates and near chance WITHOUT them, they
   are one family and separating them was never a behavioural task.

Usage:
    python3 train_family.py --splits data_out/splits --outdir reports/family
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from common import FEATURES, HGB_PARAMS
from train_model import evaluate, load_split, plot_cm, stratified_subsample

# --------------------------------------------------------------------------
# class -> family mapping (justified by the ablation + fingerprints)
# --------------------------------------------------------------------------
FAMILY = {
    "Benign": "Benign",
    "DDOS attack-HOIC": "Flood",
    "DDoS attacks-LOIC-HTTP": "Flood",
    "DoS attacks-Hulk": "Flood",
    "Bot": "Bot",
    "SSH-Bruteforce": "BruteForce",
}
FAMILY_ORDER = ["Benign", "Flood", "Bot", "BruteForce"]

# per-tool packet-template columns (see fingerprints.py output)
TOOL_GROUP = [
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Seg Size Min",
    "Fwd Header Len", "Pkt Size Avg",
]
BEHAVIOURAL = [f for f in FEATURES if f not in TOOL_GROUP]

FLOOD_CLASSES = ["DDOS attack-HOIC", "DDoS attacks-LOIC-HTTP", "DoS attacks-Hulk"]


def benign_fpr(r: dict):
    cm = np.array(r["confusion_matrix"])
    classes = list(r["per_class"].keys())
    bi = classes.index("Benign")
    n_ben = int(cm[bi].sum())
    return (n_ben - int(cm[bi, bi])) / n_ben, n_ben - int(cm[bi, bi]), n_ben


def fit_eval(Xtr, ytr, Xva, yva, classes, seed, n_est):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=n_est, class_weight="balanced", random_state=seed,
        **HGB_PARAMS)
    clf.fit(Xtr, ytr)
    return evaluate(clf, Xva, yva, classes, "VAL")


def pairwise_flood(Xtr, ytr, Xva, yva, cols, seed, n_est):
    """Mean pairwise balanced accuracy among the flood classes (1 = perfectly
    separable, 0.5 = indistinguishable)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score
    idx = {f: i for i, f in enumerate(FEATURES)}
    ci = [idx[c] for c in cols]
    scores = {}
    for i in range(len(FLOOD_CLASSES)):
        for j in range(i + 1, len(FLOOD_CLASSES)):
            a, b = FLOOD_CLASSES[i], FLOOD_CLASSES[j]
            mtr = np.isin(ytr, [a, b])
            mva = np.isin(yva, [a, b])
            if mtr.sum() < 50 or mva.sum() < 20:
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=n_est, class_weight="balanced",
                random_state=seed, early_stopping=True)
            clf.fit(Xtr[mtr][:, ci], ytr[mtr])
            s = balanced_accuracy_score(yva[mva], clf.predict(Xva[mva][:, ci]))
            scores[f"{a} vs {b}"] = float(s)
    return scores, (float(np.mean(list(scores.values()))) if scores else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data_out/splits")
    ap.add_argument("--outdir", default="reports/family")
    ap.add_argument("--max-train-rows", type=int, default=2_000_000)
    ap.add_argument("--max-val-rows", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    Xtr, ytr0 = load_split(os.path.join(args.splits, "train.parquet"))
    Xva, yva0 = load_split(os.path.join(args.splits, "val.parquet"))
    Xtr, ytr0 = stratified_subsample(Xtr, ytr0, args.max_train_rows, args.seed)
    Xva, yva0 = stratified_subsample(Xva, yva0, args.max_val_rows, args.seed)

    ytr = np.array([FAMILY[c] for c in ytr0])
    yva = np.array([FAMILY[c] for c in yva0])
    classes = np.array([c for c in FAMILY_ORDER
                        if (ytr == c).any() and (yva == c).any()])
    print(f"train={len(ytr):,}  val={len(yva):,}")
    print("family mapping:", json.dumps(FAMILY, indent=2))
    print(f"\ntrain family sizes: "
          f"{ {c: int((ytr==c).sum()) for c in classes} }")
    print(f"val   family sizes: "
          f"{ {c: int((yva==c).sum()) for c in classes} }")

    idx = {f: i for i, f in enumerate(FEATURES)}
    results = {}

    for variant, cols in (("all32", FEATURES), ("behavioural", BEHAVIOURAL)):
        ci = [idx[c] for c in cols]
        print(f"\n{'#'*72}\n# FAMILY MODEL [{variant}]  ({len(cols)} features)\n{'#'*72}")
        r = fit_eval(Xtr[:, ci], ytr, Xva[:, ci], yva, classes,
                     args.seed, args.n_estimators)
        fpr, fp, nben = benign_fpr(r)
        prec = {c: v["precision"] for c, v in r["per_class"].items()
                if c != "Benign"}
        print(f"  benign FPR: {fpr:.4%} ({fp:,}/{nben:,})")
        for c, v in prec.items():
            print(f"  {c:<12} precision {v:.4f}  recall {r['per_class'][c]['recall']:.4f}")
        results[variant] = {"n_features": len(cols), "features": cols,
                            "balanced_accuracy": r["balanced_accuracy"],
                            "macro_f1": r["macro_f1"],
                            "accuracy": r["accuracy"],
                            "benign_fpr": fpr,
                            "per_class": r["per_class"],
                            "confusion_matrix": r["confusion_matrix"]}
        plot_cm(np.array(r["confusion_matrix"]), r["classes"],
                os.path.join(args.outdir, f"cm_family_{variant}.png"),
                f"FAMILY level — {variant} ({len(cols)} feats)")

    # ---- pairwise flood separability -------------------------------------
    print(f"\n{'#'*72}\n# PAIRWISE FLOOD SEPARABILITY (1.0 = separable, 0.5 = same thing)\n{'#'*72}")
    pair = {}
    for variant, cols in (("all32", FEATURES), ("behavioural", BEHAVIOURAL)):
        s, mean = pairwise_flood(Xtr, ytr0, Xva, yva0, cols,
                                 args.seed, args.n_estimators)
        pair[variant] = {"pairs": s, "mean": mean}
        print(f"\n  [{variant}]  mean = {mean:.4f}")
        for k, v in s.items():
            print(f"     {k:<46} {v:.4f}")

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print("  A pair scoring ~0.50 without the tool templates is one class.")
    print("  (0.50 = chance, 1.00 = perfectly separable)\n")
    merged = []
    for k, v in pair["behavioural"]["pairs"].items():
        a = pair["all32"]["pairs"].get(k, float("nan"))
        verdict = "SAME CLASS - merge" if v < 0.60 else (
            "weak / partly separable" if v < 0.75 else "separable")
        if v < 0.60:
            merged.append(k)
        print(f"    {k:<44} all32 {a:.4f} -> behavioural {v:.4f}   {verdict}")
    b = results["behavioural"]
    a = results["all32"]
    print(f"\n  family model (all32)      : bal.acc {a['balanced_accuracy']:.6f}  "
          f"macroF1 {a['macro_f1']:.6f}  benign FPR {a['benign_fpr']:.4%}")
    print(f"  family model (behavioural): bal.acc {b['balanced_accuracy']:.6f}  "
          f"macroF1 {b['macro_f1']:.6f}  benign FPR {b['benign_fpr']:.4%}")
    print(f"  macro F1 drop from dropping templates: "
          f"{a['macro_f1']-b['macro_f1']:+.6f}")
    if merged:
        print(f"\n  -> Behaviourally INSEPARABLE pairs: {len(merged)}")
        for k in merged:
            print(f"       {k}")
        print("     These are the same class in behaviour terms; the family grouping")
        print("     is required, and the family result is the number to report.")
    else:
        print("\n  -> No flood pair collapsed to chance; the families stay distinct")
        print("     on behaviour. Report the family result with the all32 number as")
        print("     the fingerprint-dependent upper bound.")

    out = {"family_mapping": FAMILY, "family_order": list(classes),
           "tool_group": TOOL_GROUP,
           "family_model": results, "pairwise_flood": pair}
    with open(os.path.join(args.outdir, "family_report.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nelapsed {time.time()-t0:.1f}s -> {args.outdir}/family_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
