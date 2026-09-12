"""
Step 4 -- Baseline model on the cleaned, split data.

Fits a gradient-boosted tree classifier with class weights (so the rare classes
are not ignored), then reports honest per-class metrics on val and test, plus a
row-normalised confusion matrix.

Why HistGradientBoosting by default:
  * handles 8 M rows x 32 features in a few minutes on a laptop,
  * supports class_weight="balanced" natively,
  * needs no scaling and tolerates the -1 sentinel in Init*Win as an ordinary
    category value.

Usage:
    python3 train_model.py --splits data_out/splits --outdir reports/model

    # quick smoke test on a subsample
    python3 train_model.py --splits data_out/splits --max-train-rows 500000

    # full data, all rows
    python3 train_model.py --splits data_out/splits --max-train-rows 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from common import FEATURES, LABEL_COL, HGB_PARAMS


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_split(path: str) -> tuple[np.ndarray, np.ndarray]:
    pdf = pq.read_table(path, columns=FEATURES + [LABEL_COL]).to_pandas()
    X = pdf[FEATURES].astype(np.float64).to_numpy()
    y = pdf[LABEL_COL].astype(str).to_numpy()
    return X, y


def stratified_subsample(X: np.ndarray, y: np.ndarray, n: int,
                         seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Proportional per-class subsample of n rows (all classes represented)."""
    if n <= 0 or n >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    frac = n / len(y)
    keep = np.zeros(len(y), dtype=bool)
    for lab in np.unique(y):
        idx = np.flatnonzero(y == lab)
        k = max(1, int(round(len(idx) * frac)))
        k = min(k, len(idx))
        keep[rng.choice(idx, size=k, replace=False)] = True
    return X[keep], y[keep]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def evaluate(model, X, y, classes, tag: str) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix,
                                 f1_score)
    t0 = time.time()
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    bacc = balanced_accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    report = classification_report(y, pred, labels=list(classes),
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(classes))
    print(f"\n---- {tag} ({len(y):,} rows, {time.time()-t0:.1f}s) ----")
    print(f"accuracy          : {acc:.6f}")
    print(f"balanced accuracy : {bacc:.6f}   <- the honest headline number")
    print(f"macro F1          : {macro_f1:.6f}")
    print(f"\n{'class':<24}{'prec':>8}{'recall':>9}{'f1':>8}{'support':>11}")
    per_class = {}
    for c in classes:
        d = report.get(c, {})
        per_class[c] = {"precision": d.get("precision", 0.0),
                        "recall": d.get("recall", 0.0),
                        "f1": d.get("f1-score", 0.0),
                        "support": int(d.get("support", 0))}
        print(f"{c:<24}{d.get('precision',0):>8.4f}{d.get('recall',0):>9.4f}"
              f"{d.get('f1-score',0):>8.4f}{int(d.get('support',0)):>11,}")
    return {"tag": tag, "rows": int(len(y)), "accuracy": float(acc),
            "balanced_accuracy": float(bacc), "macro_f1": float(macro_f1),
            "per_class": per_class, "confusion_matrix": cm.tolist(),
            "classes": list(classes)}


def plot_cm(cm: np.ndarray, classes, path: str, tag: str) -> bool:
    """Row-normalised confusion matrix -> PNG. Returns False if no matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 3,
                                    0.8 * len(classes) + 2.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{tag} — row-normalised confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cmn[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data_out/splits")
    ap.add_argument("--outdir", default="reports/model")
    ap.add_argument("--model", choices=["hgb", "rf"], default="hgb")
    ap.add_argument("--max-train-rows", type=int, default=2_000_000,
                    help="stratified subsample of TRAIN for speed; 0 = all rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--binary", action="store_true",
                    help="also fit/evaluate a Benign-vs-Attack head")
    ap.add_argument("--importance", action="store_true",
                    help="compute permutation importance on a val subsample")
    ap.add_argument("--n-estimators", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    # ---------- load ------------------------------------------------------
    Xtr, ytr = load_split(os.path.join(args.splits, "train.parquet"))
    Xva, yva = load_split(os.path.join(args.splits, "val.parquet"))
    Xte, yte = load_split(os.path.join(args.splits, "test.parquet"))
    print(f"loaded  train={Xtr.shape}  val={Xva.shape}  test={Xte.shape}")

    Xtr, ytr = stratified_subsample(Xtr, ytr, args.max_train_rows, args.seed)
    print(f"training on {len(ytr):,} rows "
          f"({'ALL' if args.max_train_rows == 0 else 'subsample'})")
    classes = np.array(sorted(set(ytr) | set(yva) | set(yte)))
    print(f"classes ({len(classes)}): {list(classes)}")

    # ---------- fit -------------------------------------------------------
    if args.model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=args.n_estimators, class_weight="balanced",
            random_state=args.seed, verbose=0, **HGB_PARAMS)
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators, class_weight="balanced_subsample",
            n_jobs=-1, random_state=args.seed, max_features="sqrt",
            min_samples_leaf=2)

    print(f"\nfitting {args.model} (class_weight=balanced) ...")
    tfit = time.time()
    clf.fit(Xtr, ytr)
    print(f"fit done in {time.time()-tfit:.1f}s")

    # ---------- evaluate --------------------------------------------------
    results = {"model": args.model, "train_rows": int(len(ytr)),
               "features": FEATURES, "seed": args.seed,
               "class_weight": "balanced", "evaluations": []}
    results["evaluations"].append(evaluate(clf, Xva, yva, classes, "VAL"))
    results["evaluations"].append(evaluate(clf, Xte, yte, classes, "TEST"))

    for r in results["evaluations"]:
        p = os.path.join(args.outdir, f"confusion_{r['tag'].lower()}.png")
        if plot_cm(np.array(r["confusion_matrix"]), r["classes"], p,
                   f"{args.model.upper()} {r['tag']}"):
            print(f"wrote {p}")

    # ---------- optional binary head --------------------------------------
    if args.binary:
        ytr_b = np.where(ytr == "Benign", "Benign", "Attack")
        yva_b = np.where(yva == "Benign", "Benign", "Attack")
        yte_b = np.where(yte == "Benign", "Benign", "Attack")
        from sklearn.ensemble import HistGradientBoostingClassifier
        b = HistGradientBoostingClassifier(
            max_iter=args.n_estimators, class_weight="balanced",
            random_state=args.seed)
        b.fit(Xtr, ytr_b)
        rv = evaluate(b, Xva, yva_b, np.array(["Attack", "Benign"]),
                      "VAL (binary)")
        rt = evaluate(b, Xte, yte_b, np.array(["Attack", "Benign"]),
                      "TEST (binary)")
        results["binary"] = [rv, rt]
        for r in (rv, rt):
            plot_cm(np.array(r["confusion_matrix"]), r["classes"],
                    os.path.join(args.outdir,
                                 f"confusion_binary_{r['tag'].split()[0].lower()}.png"),
                    f"BINARY {r['tag']}")

    # ---------- optional importance ---------------------------------------
    if args.importance:
        from sklearn.inspection import permutation_importance
        sub = min(50_000, len(yva))
        idx = np.random.default_rng(args.seed).choice(len(yva), sub, replace=False)
        print(f"\npermutation importance on {sub:,} val rows ...")
        imp = permutation_importance(clf, Xva[idx], yva[idx], n_repeats=3,
                                     random_state=args.seed, n_jobs=-1,
                                     scoring="balanced_accuracy")
        order = np.argsort(imp.importances_mean)[::-1]
        print(f"\n{'rank':<6}{'feature':<26}{'drop in bal.acc':>16}")
        fi = []
        for r, i in enumerate(order, 1):
            fi.append({"rank": r, "feature": FEATURES[i],
                       "importance": float(imp.importances_mean[i]),
                       "std": float(imp.importances_std[i])})
            print(f"{r:<6}{FEATURES[i]:<26}{imp.importances_mean[i]:>16.6f}")
        results["permutation_importance"] = fi

    # ---------- save ------------------------------------------------------
    json.dump(results, open(os.path.join(args.outdir, "metrics.json"), "w"),
              indent=2)
    print(f"\ntotal elapsed {time.time()-t0:.1f}s -> {args.outdir}/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
