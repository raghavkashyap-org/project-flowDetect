"""
Step 4 -- train the deployable model and freeze the whole decision.

Why this is separate from train_model.py:
  train_model.py is a measuring instrument. It fits, reports, and throws the
  model away. This script produces the artefact you would actually ship:
  the fitted estimator, the exact feature list and order it expects, the label
  mapping, and the threshold -- in one file, with the threshold's provenance.

What it does:
  1. Fits the chosen estimator (default: the incumbent HGB) on the TRAIN split.
  2. Evaluates on TEST at your fixed operating threshold, at your deployment
     prior, reporting the numbers an operator sees: alerts and misses per 1,000
     flows, and cost per 1,000 flows.
  3. Saves model.joblib + model_card.json.
  4. `--score some.parquet` reuses a saved artefact to score new flows.

Honesty note on --include-val:
  tau = 0.92419 was chosen on the VALIDATION split. If you fold val into
  training, the score distribution shifts and that tau is no longer calibrated
  to this model. The flag is offered because a production model should use all
  the data it can, but it says so loudly and marks the card UNVERIFIED.

Usage:
    # train, evaluate at the chosen threshold, save
    python3 deploy_model.py --splits data_out/splits --outdir reports/deploy \
        --threshold 0.92419

    # score new traffic with the saved artefact
    python3 deploy_model.py --load reports/deploy/model.joblib \
        --score new_flows.parquet --out predictions.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common import FEATURES, LABEL_COL, _XGBFamily, HGB_PARAMS
from train_model import load_split, stratified_subsample
from train_family import FAMILY, FAMILY_ORDER

TOOL_GROUP = [
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Seg Size Min",
    "Fwd Header Len", "Pkt Size Avg",
]
BEHAVIOURAL = [f for f in FEATURES if f not in TOOL_GROUP]
ARTIFACT_VERSION = 1


def build_estimator(kind: str, n_estimators: int, seed: int):
    if kind == "xgboost":
        # sample weights stand in for class_weight, which XGBoost's sklearn
        # API does not expose for multiclass.
        return _XGBFamily(
            n_estimators=n_estimators, learning_rate=0.1, max_depth=6,
            tree_method="hist", eval_metric="mlogloss",
            n_jobs=-1, random_state=seed, verbosity=0)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.1, num_leaves=31,
            class_weight="balanced", n_jobs=-1, random_state=seed, verbose=-1)
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=n_estimators, class_weight="balanced",
            random_state=seed, **HGB_PARAMS)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    raise SystemExit(f"unknown --model {kind}")


def gate(scores: np.ndarray, thr: float) -> np.ndarray:
    """The deployment rule: alert when score = 1 - P(Benign) >= tau."""
    return scores >= thr


def evaluate(y_true: np.ndarray, scores: np.ndarray, thr: float,
             prior: float, cost_fp: float, cost_fn: float) -> dict:
    """Operator-facing metrics at one threshold, on a labelled split."""
    is_att = y_true != "Benign"
    flag = gate(scores, thr)
    n_att, n_ben = int(is_att.sum()), int((~is_att).sum())
    tp = int((flag & is_att).sum())
    fp = int((flag & ~is_att).sum())
    fn = n_att - tp
    fpr = fp / n_ben if n_ben else 0.0
    recall = tp / n_att if n_att else 0.0
    att_1k, ben_1k = 1000.0 * prior, 1000.0 * (1.0 - prior)
    miss_1k, fp_1k = att_1k * (1 - recall), ben_1k * fpr
    tp_1k = att_1k * recall
    # Precision at the DEPLOYMENT prior, not the split's class balance. The
    # splits are ~10% attack; production is assumed 0.1%. Reporting tp/(tp+fp)
    # here would claim ~97% precision where an operator really sees ~24%.
    alerts_1k = tp_1k + fp_1k
    prec_prior = tp_1k / alerts_1k if alerts_1k > 0 else 0.0

    per_family = {}
    for fam in FAMILY_ORDER:
        m = y_true == fam
        if not m.any():
            continue
        if fam == "Benign":
            per_family[fam] = {"n": int(m.sum()), "false_alarms": int(flag[m].sum()),
                               "false_alarm_rate": float(flag[m].mean())}
        else:
            per_family[fam] = {"n": int(m.sum()), "caught": int(flag[m].sum()),
                               "recall": float(flag[m].mean()),
                               "missed": int((~flag[m]).sum())}
    return {"threshold": thr, "n_attack": n_att, "n_benign": n_ben,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "fpr": fpr, "recall": recall,
            "precision": prec_prior,
            "precision_dataset_balance": tp / (tp + fp) if (tp + fp) else 0.0,
            "alerts_per_1k": alerts_1k,
            "misses_per_1k": miss_1k,
            "cost_per_1k": cost_fn * miss_1k + cost_fp * fp_1k,
            "per_family": per_family}


def print_eval(tag: str, r: dict, prior: float, cost_fn: float, cost_fp: float) -> None:
    print(f"\n{'-' * 70}\n{tag}   (prior {prior:g}, miss {cost_fn:g}x a false alarm)")
    print("-" * 70)
    print(f"  threshold      : {r['threshold']:.5f}")
    print(f"  attack / benign: {r['n_attack']:,} / {r['n_benign']:,}")
    print(f"  TP / FP / FN   : {r['true_positives']:,} / "
          f"{r['false_positives']:,} / {r['false_negatives']:,}")
    print(f"  FPR            : {r['fpr']:.4%}")
    print(f"  recall         : {r['recall']:.4%}")
    print(f"  precision      : {r['precision']:.4%}   at prior {prior:g} "
          f"(the operator's number)")
    print(f"                   {r['precision_dataset_balance']:.4%}   at this "
          f"split's {r['n_attack'] / (r['n_attack'] + r['n_benign']):.2%} attack "
          f"balance (do NOT quote this)")
    print(f"  workload       : {r['alerts_per_1k']:.2f} alerts and "
          f"{r['misses_per_1k']:.4f} misses per 1,000 flows")
    print(f"  cost           : {r['cost_per_1k']:.2f} per 1,000 flows")
    print(f"  per family:")
    for fam, d in r["per_family"].items():
        if fam == "Benign":
            print(f"    {fam:12s} n={d['n']:>8,}  false alarms "
                  f"{d['false_alarms']:>7,}  ({d['false_alarm_rate']:.4%})")
        else:
            print(f"    {fam:12s} n={d['n']:>8,}  caught {d['caught']:>8,}  "
                  f"recall {d['recall']:.4%}  missed {d['missed']:>6,}")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="data_out/splits")
    ap.add_argument("--outdir", default="reports/deploy")
    ap.add_argument("--model", choices=["xgboost", "lightgbm", "hgb", "rf"], default="hgb")
    ap.add_argument("--features", choices=["behavioural", "all32"],
                    default="behavioural")
    ap.add_argument("--threshold", type=float, default=0.92419,
                    help="gate threshold tau (default 0.92419, the tuned "
                         "operating point)")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="cap training rows (0 = use all of them)")
    ap.add_argument("--include-val", action="store_true",
                    help="fold validation into training for the shipped model; "
                         "tau is then NOT calibrated to it")
    ap.add_argument("--prior", type=float, default=0.001)
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--cost-fn", type=float, default=100.0)
    # scoring mode
    ap.add_argument("--load", default="", help="saved model.joblib to reuse")
    ap.add_argument("--score", default="", help="parquet of new flows to score")
    ap.add_argument("--out", default="predictions.parquet")
    args = ap.parse_args()

    import joblib

    # ---------------- scoring mode -----------------------------------------
    if args.load:
        if not args.score:
            raise SystemExit("--load needs --score")
        import pandas as pd
        art = joblib.load(args.load)
        card = art["card"]
        cols = card["features"]
        df = pd.read_parquet(args.score)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"input is missing required columns: {missing}")
        X = df[cols].astype(np.float64).to_numpy()
        proba = art["model"].predict_proba(X)
        bi = list(art["model"].classes_).index("Benign")
        scores = 1.0 - proba[:, bi]
        df = df.copy()
        df["attack_score"] = scores
        df["family"] = art["model"].predict(X)
        df["alert"] = gate(scores, card["threshold"])
        df.to_parquet(args.out, index=False)
        n = len(df)
        print(f"scored {n:,} flows with {card['model_kind']} "
              f"({card['feature_set']}, tau={card['threshold']:.5f})")
        print(f"  alerts: {int(df['alert'].sum()):,} "
              f"({df['alert'].mean():.4%})")
        print(f"  family mix: {df['family'].value_counts().to_dict()}")
        print(f"wrote {args.out}")
        return 0

    # ---------------- training mode ----------------------------------------
    cols = FEATURES if args.features == "all32" else BEHAVIOURAL
    ci = [FEATURES.index(c) for c in cols]
    os.makedirs(args.outdir, exist_ok=True)

    Xtr, ytr0 = load_split(os.path.join(args.splits, "train.parquet"))
    Xte, yte0 = load_split(os.path.join(args.splits, "test.parquet"))
    ytr = np.array([FAMILY[v] for v in ytr0])
    yte = np.array([FAMILY[v] for v in yte0])

    tau_status = "verified on held-out test"
    if args.include_val:
        Xva, yva0 = load_split(os.path.join(args.splits, "val.parquet"))
        yva = np.array([FAMILY[v] for v in yva0])
        Xtr = np.vstack([Xtr, Xva])
        ytr = np.concatenate([ytr, yva])
        tau_status = ("UNVERIFIED -- val was folded into training, so tau is "
                      "not calibrated to this fit")
        print("WARNING: --include-val folds validation into training.")
        print(f"         tau={args.threshold:.5f} was chosen ON that validation")
        print("         split, so it is no longer calibrated to this model.")
        print("         Re-run tune_threshold.py before trusting the threshold.")

    if args.max_train_rows and len(ytr) > args.max_train_rows:
        Xtr, ytr = stratified_subsample(Xtr, ytr, args.max_train_rows)
    Xtr, Xte = Xtr[:, ci], Xte[:, ci]

    print(f"features : {args.features} ({len(cols)})")
    print(f"model    : {args.model} ({args.n_estimators} estimators, "
          f"seed {args.seed})")
    print(f"train    : {len(ytr):,} rows   test: {len(yte):,} rows")
    print(f"tau      : {args.threshold:.5f}   [{tau_status}]")

    t0 = time.time()
    est = build_estimator(args.model, args.n_estimators, args.seed)
    if args.model == "xgboost":
        # XGBoost has no multiclass class_weight; use equivalent sample weights.
        classes, counts = np.unique(ytr, return_counts=True)
        w = len(ytr) / (len(classes) * counts.astype(np.float64))
        est.fit(Xtr, ytr, sample_weight=w[np.searchsorted(classes, ytr)])
    else:
        est.fit(Xtr, ytr)
    fit_s = time.time() - t0
    print(f"fitted in {fit_s:.1f}s")

    proba = est.predict_proba(Xte)
    scores = 1.0 - proba[:, list(est.classes_).index("Benign")]
    r = evaluate(yte, scores, args.threshold,
                 args.prior, args.cost_fp, args.cost_fn)
    print_eval("TEST at the operating point", r,
               args.prior, args.cost_fn, args.cost_fp)

    card = {
        "artifact_version": ARTIFACT_VERSION,
        "model_kind": args.model, "n_estimators": args.n_estimators,
        "seed": args.seed, "feature_set": args.features,
        "n_features": len(cols), "features": cols,
        "label_column": LABEL_COL, "families": FAMILY_ORDER,
        "gate": "alert when (1 - P(Benign)) >= threshold",
        "threshold": args.threshold, "threshold_status": tau_status,
        "trained_on_val": bool(args.include_val),
        "n_train_rows": int(len(ytr)), "fit_seconds": round(fit_s, 1),
        "deployment": {"prior": args.prior, "cost_fp": args.cost_fp,
                       "cost_fn": args.cost_fn},
        "test_metrics": r,
        "known_limitation": ("each attack class appears on a single capture day "
                             "in CICIDS2017, so cross-session generalisation is "
                             "untested"),
    }
    mp = os.path.join(args.outdir, "model.joblib")
    cp = os.path.join(args.outdir, "model_card.json")
    joblib.dump({"model": est, "card": card}, mp)
    with open(cp, "w") as fh:
        json.dump(card, fh, indent=2)
    print(f"\nwrote {mp}\nwrote {cp}")
    print("\nTo score new traffic:")
    print(f"  python3 deploy_model.py --load {mp} --score new.parquet "
          f"--out preds.parquet")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
