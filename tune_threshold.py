"""
Threshold tuning -- turn the class-weight bias into a chosen operating point.

Why this is needed:
  The baseline uses class_weight="balanced", which deliberately over-predicts
  minority classes. On the family model that produced 3,834 false alarms vs
  only 180 missed attacks (21:1). Recall was already near ceiling, so those
  weights bought almost nothing and cost a lot of analyst time.

How it works:
  An IDS decision is really two steps:
      1. GATE   -- alert at all?  score = 1 - P(Benign)
      2. LABEL  -- among alerts, which family is most likely?
  Only step 1 needs a threshold. This script sweeps it exhaustively (using a
  sort + cumulative sum, so every candidate threshold is evaluated exactly --
  no coarse grid), reports the full precision/recall/FPR trade-off, and picks
  the operating point that meets a target false-positive rate.

Methodology:
  The threshold is CHOSEN on validation and REPORTED on test. Never pick the
  operating point on the data you report it from.

Usage:
    python3 tune_threshold.py --splits data_out/splits --outdir reports/threshold

    # target a 0.1% false-positive rate (default)
    python3 tune_threshold.py --target-fpr 0.001

    # unweighted model: probabilities closer to the true prior
    python3 tune_threshold.py --no-class-weights

    # compare several operating points side by side
    python3 tune_threshold.py --target-fpr 0.001,0.005,0.01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from common import FEATURES, HGB_PARAMS
from train_model import load_split, stratified_subsample
from train_family import FAMILY, FAMILY_ORDER

TOOL_GROUP = [
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Seg Size Min",
    "Fwd Header Len", "Pkt Size Avg",
]
BEHAVIOURAL = [f for f in FEATURES if f not in TOOL_GROUP]


# --------------------------------------------------------------------------
def sweep(scores: np.ndarray, is_attack: np.ndarray) -> dict:
    """Exact precision / recall / FPR at EVERY achievable threshold.

    Sort rows by descending attack score: as the threshold falls we admit rows
    one at a time, so cumulative sums give TP and FP without re-scanning.
    """
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    a = is_attack[order].astype(np.int64)

    tp = np.cumsum(a)                 # admitted rows that really are attacks
    fp = np.cumsum(1 - a)             # admitted rows that are really benign
    n_pos = int(a.sum())
    n_neg = int(len(a) - n_pos)

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = tp / n_pos if n_pos else np.zeros_like(tp, dtype=float)
        fpr = fp / n_neg if n_neg else np.zeros_like(fp, dtype=float)
        prec = np.where((tp + fp) > 0, tp / np.maximum(tp + fp, 1), 1.0)
        f1 = np.where((prec + recall) > 0,
                      2 * prec * recall / np.maximum(prec + recall, 1e-12), 0.0)
    return {"threshold": s, "recall": np.asarray(recall, dtype=float),
            "fpr": np.asarray(fpr, dtype=float), "precision": np.asarray(prec),
            "f1": np.asarray(f1), "tp": tp, "fp": fp,
            "n_pos": n_pos, "n_neg": n_neg}


def pick_for_fpr(sw: dict, target_fpr: float) -> dict:
    """Threshold whose FPR is the largest value still <= target_fpr.

    If no threshold is that strict, fall back to the strictest available.
    """
    fpr = sw["fpr"]
    ok = np.flatnonzero(fpr <= target_fpr)
    i = int(ok[-1]) if len(ok) else 0
    return {"index": i, "threshold": float(sw["threshold"][i]),
            "fpr": float(fpr[i]), "recall": float(sw["recall"][i]),
            "precision": float(sw["precision"][i]), "f1": float(sw["f1"][i]),
            "tp": int(sw["tp"][i]), "fp": int(sw["fp"][i]),
            "achieved_target": bool(fpr[i] <= target_fpr)}


def apply_threshold(scores, pred_proba_classes, classes, thr):
    """Gate at thr, then label alerts by argmax among NON-benign classes."""
    alert = scores >= thr
    out = np.array(["Benign"] * len(scores), dtype=object)
    if alert.any() and "Benign" in classes:
        bi = classes.index("Benign")
        nonb = [j for j in range(len(classes)) if j != bi]
        sub = pred_proba_classes[np.ix_(np.flatnonzero(alert), nonb)]
        pick = np.asarray(nonb)[sub.argmax(axis=1)]
        out[np.flatnonzero(alert)] = [classes[j] for j in pick]
    based = {c: i for i, c in enumerate(classes)}
    return out, alert, based


def report_at(y_true, y_pred, classes, thr, tag):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    labels = list(classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    bi = labels.index("Benign")
    n_ben = int(cm[bi].sum())
    fp = n_ben - int(cm[bi, bi])
    print(f"\n---- {tag}  (threshold = {thr:.6f}) ----")
    print(f"  benign FPR : {fp/n_ben:.4%}  ({fp:,}/{n_ben:,})")
    print(f"\n  {'class':<12}{'prec':>9}{'recall':>9}{'f1':>8}{'support':>10}")
    for i, c in enumerate(labels):
        print(f"  {c:<12}{p[i]:>9.4f}{r[i]:>9.4f}{f[i]:>8.4f}{int(s[i]):>10,}")
    return {"tag": tag, "threshold": float(thr),
            "benign_fpr": float(fp / n_ben if n_ben else 0.0),
            "false_positives": int(fp), "benign_rows": n_ben,
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]),
                              "f1": float(f[i]), "support": int(s[i])}
                          for i, c in enumerate(labels)},
            "confusion_matrix": cm.tolist(), "classes": labels}


def plot_curves(sw_val, sw_test, picks, path, target_fprs,
                cost_best=None, costs=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes

    # left: ROC-like  recall vs FPR
    for sw, lab, st in ((sw_val, "validation", "-"), (sw_test, "test", "--")):
        ax1.plot(sw["fpr"], sw["recall"], st, lw=1.6, label=lab)
    for t in target_fprs:
        ax1.axvline(t, color="grey", ls=":", lw=1)
        ax1.text(t, 0.02, f"{t:.3%}", rotation=90, fontsize=7, color="grey")
    if picks:
        ax1.scatter([p["fpr"] for p in picks], [p["recall"] for p in picks],
                    s=46, c="crimson", zorder=5, label="chosen operating point")
    ax1.set_xscale("log")
    ax1.set_xlabel("false-positive rate (log)")
    ax1.set_ylabel("attack detection rate (recall)")
    ax1.set_title("Detection trade-off")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)

    # right: metrics vs threshold
    ax2.plot(sw_val["threshold"], sw_val["precision"], lw=1.5, label="precision")
    ax2.plot(sw_val["threshold"], sw_val["recall"], lw=1.5, label="recall")
    ax2.plot(sw_val["threshold"], sw_val["f1"], lw=1.5, label="F1")
    ax2.plot(sw_val["threshold"], sw_val["fpr"], lw=1.5, label="benign FPR")
    for p in picks:
        ax2.axvline(p["threshold"], color="crimson", ls="--", lw=1)
    ax2.set_xlabel("alert threshold on  1 − P(Benign)")
    ax2.set_ylabel("metric")
    ax2.set_title("Metrics vs threshold (validation)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9)

    # right: expected cost vs threshold
    if costs is not None:
        ax3.plot(sw_val["threshold"], costs, lw=1.5, color="darkorange")
        if cost_best is not None:
            ax3.axvline(cost_best["threshold"], color="crimson", ls="--", lw=1,
                        label=f"optimal (t={cost_best['threshold']:.3f})")
            ax3.legend(fontsize=9)
        ax3.set_xlabel("alert threshold on  1 - P(Benign)")
        ax3.set_ylabel("expected cost (weighted, arbitrary units)")
        ax3.set_title("Cost-optimal operating point (validation)")
        ax3.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data_out/splits")
    ap.add_argument("--outdir", default="reports/threshold")
    ap.add_argument("--features", choices=["behavioural", "all32"],
                    default="behavioural")
    ap.add_argument("--target-fpr", default="0.001",
                    help="comma-separated target false-positive rates")
    ap.add_argument("--no-class-weights", dest="balanced",
                    action="store_false", default=True,
                    help="train without class weights (probabilities closer to "
                         "the true prior; pairs well with threshold tuning)")
    ap.add_argument("--max-train-rows", type=int, default=2_000_000)
    ap.add_argument("--max-eval-rows", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--prior", type=float, default=0.0,
                    help="deployment attack rate to cost against, as a "
                         "fraction (e.g. 0.0001 = 1 attack per 10,000 flows). "
                         "Default 0 = use the dataset's own class balance, "
                         "which makes false alarms look far cheaper than they "
                         "are in real traffic.")
    ap.add_argument("--cost-fp", type=float, default=1.0,
                    help="relative cost of one false alarm (default 1)")
    ap.add_argument("--cost-fn", type=float, default=10.0,
                    help="relative cost of one missed attack (default 10; a "
                         "miss is normally worse than a false alarm)")
    args = ap.parse_args()

    targets = [float(x) for x in args.target_fpr.split(",") if x.strip()]
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    cols = BEHAVIOURAL if args.features == "behavioural" else FEATURES
    idx = {f: i for i, f in enumerate(FEATURES)}
    ci = [idx[c] for c in cols]

    Xtr, ytr0 = load_split(os.path.join(args.splits, "train.parquet"))
    Xva, yva0 = load_split(os.path.join(args.splits, "val.parquet"))
    Xte, yte0 = load_split(os.path.join(args.splits, "test.parquet"))
    Xtr, ytr0 = stratified_subsample(Xtr, ytr0, args.max_train_rows, args.seed)
    Xva, yva0 = stratified_subsample(Xva, yva0, args.max_eval_rows, args.seed)
    Xte, yte0 = stratified_subsample(Xte, yte0, args.max_eval_rows, args.seed)

    ytr = np.array([FAMILY[c] for c in ytr0])
    yva = np.array([FAMILY[c] for c in yva0])
    yte = np.array([FAMILY[c] for c in yte0])
    classes = [c for c in FAMILY_ORDER if (ytr == c).any()]
    print(f"features  : {args.features} ({len(cols)})")
    print(f"weights   : {'balanced' if args.balanced else 'none'}")
    print(f"train/val/test = {len(ytr):,} / {len(yva):,} / {len(yte):,}")
    print(f"targets   : {[f'{t:.4%}' for t in targets]}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=args.n_estimators,
        class_weight="balanced" if args.balanced else None,
        random_state=args.seed, **HGB_PARAMS)
    print("\nfitting ...")
    clf.fit(Xtr[:, ci], ytr)
    clf_classes = list(clf.classes_)
    bi = clf_classes.index("Benign")

    def scores_labels(X, y):
        pr = clf.predict_proba(X[:, ci])
        sc = 1.0 - pr[:, bi]
        return sc, (y != "Benign")

    sv, av = scores_labels(Xva, yva)
    st, at = scores_labels(Xte, yte)
    sw_val, sw_test = sweep(sv, av), sweep(st, at)

    # default (untuned) behaviour for comparison -- the model's own argmax
    from sklearn.metrics import confusion_matrix as _cm
    base_pred = clf.predict(Xva[:, ci])
    _b = _cm(yva, base_pred, labels=classes)
    _bi = classes.index("Benign")
    _nben = int(_b[_bi].sum())
    base_fp = _nben - int(_b[_bi, _bi])
    base_recall = float((base_pred != "Benign")[yva != "Benign"].mean())
    print(f"\n[reference] UNTUNED model (argmax, no threshold):")
    print(f"            benign FPR {base_fp/_nben:.4%} ({base_fp:,}/{_nben:,})"
          f"   attack recall {base_recall:.4%}")

    picks, rows = [], []
    print("\n" + "=" * 70)
    print("OPERATING POINTS (threshold chosen on VALIDATION)")
    print("=" * 70)
    print(f"{'target FPR':>11}{'thresh':>10}{'FPR(val)':>11}{'recall(val)':>13}"
          f"{'prec(val)':>11}{'F1(val)':>10}")
    for t in targets:
        p = pick_for_fpr(sw_val, t)
        p["target_fpr"] = t
        picks.append(p)
        print(f"{t:>11.4%}{p['threshold']:>10.5f}{p['fpr']:>11.4%}"
              f"{p['recall']:>13.4%}{p['precision']:>11.4f}{p['f1']:>10.4f}")
        rows.append(p)

    # ---- cost-optimal operating point ------------------------------------
    # expected cost = cost_fn * FN + cost_fp * FP  (cost_fn should exceed
    # cost_fp: a missed attack is normally worse than a false alarm)
    #
    # WEIGHTING MATTERS. Counting raw validation rows costs the model at the
    # dataset's own class balance (here ~9.8% attack). Real traffic is orders
    # of magnitude more benign, so an unweighted cost makes false alarms look
    # far cheaper than they are and drags the optimum towards a permissive
    # threshold. --prior reweights both error types to a deployment attack
    # rate; the weights are normalised so "per 1,000 rows" still means per
    # 1,000 flows of deployment traffic.
    fn = sw_val["n_pos"] - sw_val["tp"]
    p_val = sw_val["n_pos"] / len(sv)
    if args.prior > 0.0:
        if not 0.0 < args.prior < 1.0:
            raise SystemExit("--prior must be in (0, 1)")
        w_att = args.prior / p_val
        w_ben = (1.0 - args.prior) / (1.0 - p_val)
        prior_note = f"deployment prior {args.prior:g}"
    else:
        w_att = w_ben = 1.0
        prior_note = f"dataset balance ({p_val:.2%} attack)"
    fn_w = fn * w_att
    fp_w = sw_val["fp"] * w_ben
    cost = args.cost_fn * fn_w + args.cost_fp * fp_w
    ci_best = int(cost.argmin())
    cost_best = {
        "threshold": float(sw_val["threshold"][ci_best]),
        "fpr": float(sw_val["fpr"][ci_best]),
        "recall": float(sw_val["recall"][ci_best]),
        "precision": float(sw_val["precision"][ci_best]),
        "f1": float(sw_val["f1"][ci_best]),
        "false_positives": int(sw_val["fp"][ci_best]),
        "false_negatives": int(fn[ci_best]),
        "weighted_false_positives": float(fp_w[ci_best]),
        "weighted_false_negatives": float(fn_w[ci_best]),
        "alerts_per_1k_flows": float(
            1000 * (sw_val["tp"][ci_best] * w_att + fp_w[ci_best]) / len(sv)),
        "misses_per_1k_flows": float(1000 * fn_w[ci_best] / len(sv)),
        "prior": float(args.prior),
        "dataset_attack_rate": float(p_val),
        "total_cost": float(cost[ci_best]),
        "cost_per_1k_rows": float(1000 * cost[ci_best] / len(sv)),
    }
    print("\n" + "=" * 70)
    print(f"COST-OPTIMAL POINT  (cost_fn={args.cost_fn:g} x cost_fp="
          f"{args.cost_fp:g}, {prior_note})")
    print("=" * 70)
    print(f"  threshold   : {cost_best['threshold']:.5f}")
    print(f"  benign FPR  : {cost_best['fpr']:.4%}")
    print(f"  recall      : {cost_best['recall']:.4%}")
    print(f"  FP / FN     : {cost_best['false_positives']:,} / "
          f"{cost_best['false_negatives']:,}")
    if args.prior > 0.0:
        print(f"  workload    : {cost_best['alerts_per_1k_flows']:.2f} alerts and "
              f"{cost_best['misses_per_1k_flows']:.3f} misses per 1,000 flows")
    print(f"  cost        : {cost_best['cost_per_1k_rows']:.2f} per 1,000 rows")

    # ---- the honest step: report on TEST with the VAL-chosen threshold ----
    out_test = []
    for t, p in zip(targets, picks):
        pred, _, _ = apply_threshold(st, clf.predict_proba(Xte[:, ci]),
                                     clf_classes, p["threshold"])
        r = report_at(yte, pred, classes, p["threshold"],
                      f"TEST @ target FPR {t:.4%}")
        r["target_fpr"] = t
        r["threshold"] = float(p["threshold"])
        out_test.append(r)

    pk = os.path.join(args.outdir, "threshold_curves.png")
    if plot_curves(sw_val, sw_test, picks, pk, targets,
                   cost_best=cost_best, costs=cost):
        print(f"\nwrote {pk}")

    # The JSON keeps a downsampled curve (small, readable, fine for plotting).
    # But re-optimising a downsampled curve picks a NEARBY threshold, not the
    # true optimum -- so the full sweep is also written to a compressed sidecar
    # that threshold_decision.py prefers when it is present.
    step = max(1, len(sw_val["threshold"]) // 4000)
    curve = {"threshold": sw_val["threshold"][::step].tolist(),
             "fpr": sw_val["fpr"][::step].tolist(),
             "recall": sw_val["recall"][::step].tolist(),
             "precision": sw_val["precision"][::step].tolist(),
             "f1": sw_val["f1"][::step].tolist()}
    np.savez_compressed(
        os.path.join(args.outdir, "threshold_sweep.npz"),
        threshold=sw_val["threshold"], fpr=sw_val["fpr"],
        recall=sw_val["recall"], precision=sw_val["precision"],
        f1=sw_val["f1"], tp=sw_val["tp"], fp=sw_val["fp"],
        n_pos=sw_val["n_pos"], n_neg=sw_val["n_neg"])

    summary = {
        "features": args.features, "n_features": len(cols),
        "class_weight": "balanced" if args.balanced else None,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "untuned_argmax": {"benign_fpr": float(base_fp / _nben),
                           "false_positives": int(base_fp),
                           "benign_rows": int(_nben),
                           "attack_recall": base_recall},
        "cost_model": {"cost_false_positive": args.cost_fp,
                       "cost_false_negative": args.cost_fn,
                       "prior": args.prior,
                       "dataset_attack_rate": cost_best["dataset_attack_rate"],
                       "note": ("costs are weighted to `prior` when set; "
                                "otherwise they use the dataset's own class "
                                "balance, which understates the price of a "
                                "false alarm in real traffic")},
        "cost_optimal": cost_best,
        "chosen_on_validation": rows,
        "test_at_chosen_threshold": out_test,
        "sweep_points_evaluated": int(len(sw_val["threshold"])),
        "sweep_points_in_json": len(curve["threshold"]),
        "sweep_full_path": "threshold_sweep.npz",
        "curve_validation": curve,
    }
    with open(os.path.join(args.outdir, "threshold_report.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nelapsed {time.time()-t0:.1f}s -> "
          f"{args.outdir}/threshold_report.json")
    print(f"  full sweep ({len(sw_val['threshold']):,} thresholds) -> "
          f"{args.outdir}/threshold_sweep.npz")
    print(f"  (the JSON keeps a {len(curve['threshold']):,}-point downsample; "
          f"threshold_decision.py prefers the .npz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
