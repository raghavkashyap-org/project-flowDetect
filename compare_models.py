"""
Step 3 -- compare candidate models AT YOUR OPERATING POINT.

The trap this avoids:
  The behavioural family model already reaches balanced accuracy 0.9949. At that
  level, accuracy and macro-F1 are SATURATED -- every reasonable learner lands
  within a few thousandths, so comparing algorithms by accuracy measures noise,
  not quality. Picking a model that way is how you choose the wrong one.

  Your need is already pinned down: prior ~1 attack per 1,000 flows, a miss
  costs ~100x a false alarm, and the cost-optimal gate sits near FPR 0.3%.
  So the metric that actually discriminates is

      RECALL AT A FIXED FALSE-POSITIVE RATE

  Everything here is ranked by that. Flood recall is reported separately
  because ~98% of the misses at a strict gate are Flood.

Fairness:
  Every candidate is given the SAME class-imbalance treatment. sklearn estimators
  take class_weight; XGBoost's sklearn API has no multiclass class_weight, so it
  is fitted with equivalent per-sample weights instead. Without that, XGBoost
  would be compared on unequal footing and would look artificially weak.

Usage:
    python3 compare_models.py --splits data_out/splits --outdir reports/compare

    # rank at the FPR your threshold tuning chose
    python3 compare_models.py --from-report reports/threshold/threshold_report.json

    # stage the run: the two fast gradient boosters first, the forests later
    python3 compare_models.py --models "LightGBM,HGB (current)"
    python3 compare_models.py --models "RandomForest,ExtraTrees" --max-train-rows 1000000

    # list what will run and what is missing, without fitting anything
    python3 compare_models.py --list
"""
from __future__ import annotations

import argparse
import json
import os
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

# The model already deployed in tune_threshold.py / train_family.py. Verdicts
# are made against THIS, not against the weakest thing in the list.
INCUMBENT = "HGB (current)"
# Deliberately weak baselines kept to sanity-check the metric. They are expected
# to lose and must not drive the "is it worth switching" verdict.
REFERENCE_MODELS = {"LogReg"}

# Row count above which the bagged forests get an explicit runtime warning.
SLOW_WARN_ROWS = 500_000

# What to pip install if a candidate is unavailable. Kept next to the candidate
# so the message is actionable rather than just a shrug.
OPTIONAL_DEPS = {
    "LightGBM": "lightgbm",
    "XGBoost": "xgboost",
    "CatBoost": "catboost",
}


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
class _XGBMulti:
    """XGBoost's sklearn API requires integer labels 0..K-1 for multiclass.

    Every other candidate here accepts the string family labels directly, so
    XGBoost gets wrapped: labels are encoded on fit and `classes_` is exposed in
    the same sorted order XGBoost uses internally, which keeps
    `list(est.classes_).index("Benign")` valid for it too.
    """

    def __init__(self, **kwargs):
        self._kw = kwargs

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier
        self.classes_, y_int = np.unique(y, return_inverse=True)
        self.est_ = XGBClassifier(**self._kw)
        self.est_.fit(X, y_int, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(X)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for the b/c discordant counts.

    `b` = attacks the challenger catches and the incumbent misses, `c` = the
    reverse. A recall gap on its own says nothing about whether the difference
    is real; this tests whether the disagreements are one-sided.
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * 2.0 / (2 ** n)
    return min(1.0, tail)


# Run-config fields that must match before two runs may share one report file.
MERGE_KEYS = ("features", "target_fpr", "n_train", "n_val", "n_test",
              "prior", "cost_fp", "cost_fn")


def write_report(path: str, new: dict, tag: str) -> tuple[str, str]:
    """Write `new`, merging into an existing report only if the config matches.

    Staged runs (--models A,B then --models C) would otherwise silently
    overwrite each other and leave a report containing only the last batch.
    Returns (path_written, what_happened).
    """
    import time as _t
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh)
        diff = {k: (old.get(k), new.get(k)) for k in MERGE_KEYS
                if old.get(k) != new.get(k)}
        if diff:
            alt = path[:-5] + (f"_{tag}" if tag else
                               f"_{_t.strftime('%Y%m%d-%H%M%S')}") + ".json"
            detail = ", ".join(f"{k}: {a} != {b}" for k, (a, b) in diff.items())
            print(f"\nREFUSING to overwrite {path}: run config differs ({detail}).")
            print(f"  A different train size or FPR target is a different")
            print(f"  experiment; merging them into one ranking would be a lie.")
            print(f"  Wrote {alt} instead.")
            path = alt
        else:
            by = {r["model"]: r for r in old.get("ranking", [])}
            kept = [m for m in by if m not in {r["model"] for r in new["ranking"]}]
            for r in new["ranking"]:
                by[r["model"]] = r
            new["ranking"] = sorted(by.values(),
                                    key=lambda r: -r["test"]["recall"])
            new["merged_from"] = sorted(set(old.get("merged_from", []) + kept))
            print(f"\nmerged with {len(kept)} model(s) already in {path}: "
                  f"{kept if kept else 'none'}")
    with open(path, "w") as fh:
        json.dump(new, fh, indent=2)
    return path, "merged" if "merged_from" in new else "written"


def candidates(n_jobs: int = -1) -> dict:
    """Estimators to compare.

    Returns {name: (estimator, needs_sample_weight)}. sklearn estimators handle
    imbalance through class_weight; XGBoost needs explicit sample weights.
    """
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                 RandomForestClassifier, ExtraTreesClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out: dict[str, tuple[object, bool]] = {
        # the incumbent
        "HGB (current)": (HistGradientBoostingClassifier(
            max_iter=300, class_weight="balanced", random_state=42,
            **HGB_PARAMS), False),
        # same family, different depth/regularisation
        "HGB deep": (HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=40, class_weight="balanced", random_state=42), False),
        # leaf-wise boosting: the usual winner on tabular data, and fast
        # "RandomForest"/"ExtraTrees" are appended below so the cheap models run first
    }
    try:
        from lightgbm import LGBMClassifier
        out["LightGBM"] = (LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            class_weight="balanced", n_jobs=n_jobs, random_state=42,
            verbose=-1), False)
    except ImportError:
        pass
    try:
        import xgboost  # noqa: F401  -- presence check only; wrapped below
        out["XGBoost"] = (_XGBMulti(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            n_jobs=n_jobs, random_state=42, eval_metric="mlogloss",
            tree_method="hist", verbosity=0), True)   # <- needs sample weights
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        out["CatBoost"] = (CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            auto_class_weights="Balanced", loss_function="MultiClass",
            thread_count=n_jobs if n_jobs > 0 else -1,
            random_seed=42, verbose=0, allow_writing_files=False), False)
    except ImportError:
        pass

    # bagging instead of boosting -- different bias/variance split, but slow at 2M rows
    out["RandomForest"] = (RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=n_jobs, random_state=42), False)
    out["ExtraTrees"] = (ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=n_jobs, random_state=42), False)
    # a linear reference point: if this is close, the problem is linear
    out["LogReg"] = (make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")), False)
    return out


def missing_deps() -> dict:
    """Which optional learners are unavailable, and the package that provides them."""
    import importlib
    bad = {}
    for name, pkg in OPTIONAL_DEPS.items():
        try:
            importlib.import_module(pkg)
        except ImportError:
            bad[name] = pkg
    return bad


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """sklearn class_weight='balanced' as explicit per-sample weights.

    weight_c = n_samples / (n_classes * n_samples_c), identical to
    sklearn.utils.class_weight.compute_class_weight('balanced', ...).
    """
    classes, counts = np.unique(y, return_counts=True)
    w = len(y) / (len(classes) * counts.astype(np.float64))
    return w[np.searchsorted(classes, y)].astype(np.float64)


# --------------------------------------------------------------------------
def recall_at_fpr(scores: np.ndarray, is_attack: np.ndarray,
                  target_fpr: float) -> dict:
    """Highest recall achievable while FPR stays <= target_fpr."""
    order = np.argsort(-scores, kind="mergesort")
    a = is_attack[order].astype(np.int64)
    tp = np.cumsum(a)
    fp = np.cumsum(1 - a)
    n_pos, n_neg = int(a.sum()), int(len(a) - a.sum())
    fpr = fp / n_neg
    recall = tp / n_pos
    ok = np.flatnonzero(fpr <= target_fpr)
    i = int(ok[-1]) if len(ok) else 0
    return {"threshold": float(scores[order][i]), "fpr": float(fpr[i]),
            "recall": float(recall[i]),
            "false_positives": int(fp[i]), "false_negatives": int(n_pos - tp[i])}


def eval_at(scores: np.ndarray, is_attack: np.ndarray, thr: float,
            is_flood: np.ndarray | None = None) -> dict:
    """Metrics on one split at ONE GIVEN threshold (no re-tuning).

    This is the deployment question. `recall_at_fpr` answers a different one --
    the best a model could do if you were allowed to tune on that same split.
    Reporting only the latter makes a model look better than it would run.
    """
    flag = scores >= thr
    n_pos, n_neg = int(is_attack.sum()), int((~is_attack).sum())
    tp = int((flag & is_attack).sum())
    fp = int((flag & ~is_attack).sum())
    out = {"threshold": float(thr), "fpr": fp / n_neg if n_neg else 0.0,
           "recall": tp / n_pos if n_pos else 0.0,
           "false_positives": fp, "false_negatives": n_pos - tp}
    if is_flood is not None:
        out["flood_recall"] = float(flag[is_flood].mean()) if is_flood.any() else 0.0
    return out


def deployment(res: dict, prior: float, cost_fp: float, cost_fn: float) -> dict:
    """Translate (recall, FPR) into per-1,000-deployment-flow operator numbers."""
    att_1k, ben_1k = 1000.0 * prior, 1000.0 * (1.0 - prior)
    tp, fn, fp = att_1k * res["recall"], att_1k * (1 - res["recall"]), ben_1k * res["fpr"]
    return {"alerts_per_1k": tp + fp, "misses_per_1k": fn,
            "cost_per_1k": cost_fn * fn + cost_fp * fp,
            "precision": tp / max(tp + fp, 1e-12)}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="data_out/splits")
    ap.add_argument("--outdir", default="reports/compare")
    ap.add_argument("--features", choices=["behavioural", "all32"],
                    default="behavioural")
    ap.add_argument("--at-fpr", type=float, default=0.003,
                    help="FPR to rank candidates at (default 0.003 = 0.3%%)")
    ap.add_argument("--from-report", default="",
                    help="threshold_report.json to take the target FPR from "
                         "(uses its cost-optimal point)")
    ap.add_argument("--max-train-rows", type=int, default=2_000_000)
    ap.add_argument("--max-eval-rows", type=int, default=400_000)
    ap.add_argument("--models", default="",
                    help="comma-separated subset of candidate names to run; "
                         "default is all available")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="threads per estimator (default -1 = all cores)")
    ap.add_argument("--prior", type=float, default=0.001,
                    help="deployment attack rate, for the cost columns (default 0.001)")
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--cost-fn", type=float, default=100.0)
    ap.add_argument("--list", action="store_true",
                    help="print the candidate set and exit without fitting")
    ap.add_argument("--tag", default="",
                    help="suffix for the report filename when the run config "
                         "differs from an existing report (default: timestamp)")
    args = ap.parse_args()

    all_c = candidates(args.n_jobs)
    want = [m.strip() for m in args.models.split(",") if m.strip()] if args.models \
        else list(all_c)
    unknown = [m for m in want if m not in all_c]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}\n"
                         f"available: {list(all_c)}")

    bad = missing_deps()
    if bad:
        pkgs = " ".join(sorted(set(bad.values())))
        print(f"WARNING: not installed, so these candidates will be SKIPPED:")
        for n, p in bad.items():
            print(f"  - {n:10s} (pip install {p})")
        print(f"  to include them:  pip install {pkgs}\n")

    if args.list:
        print("candidates that WILL run:")
        for n in want:
            print(f"  - {n}")
        print(f"\n{len(want)} estimator(s)")
        return 0

    cols = FEATURES if args.features == "all32" else BEHAVIOURAL
    ci = [FEATURES.index(c) for c in cols]
    os.makedirs(args.outdir, exist_ok=True)

    target_fpr = args.at_fpr
    if args.from_report:
        with open(args.from_report) as fh:
            target_fpr = float(json.load(fh)["cost_optimal"]["fpr"])
        print(f"target FPR {target_fpr:.4%} taken from {args.from_report}")

    Xtr, ytr0 = load_split(os.path.join(args.splits, "train.parquet"))
    Xva, yva0 = load_split(os.path.join(args.splits, "val.parquet"))
    Xte, yte0 = load_split(os.path.join(args.splits, "test.parquet"))
    ytr = np.array([FAMILY[v] for v in ytr0])
    yva = np.array([FAMILY[v] for v in yva0])
    yte = np.array([FAMILY[v] for v in yte0])
    Xtr, ytr = stratified_subsample(Xtr, ytr, args.max_train_rows)
    Xva, yva = stratified_subsample(Xva, yva, args.max_eval_rows)
    Xte, yte = stratified_subsample(Xte, yte, args.max_eval_rows)
    Xtr, Xva, Xte = Xtr[:, ci], Xva[:, ci], Xte[:, ci]

    a_va = yva != "Benign"
    a_te = yte != "Benign"
    w_tr = balanced_sample_weight(ytr)
    n_att_te = int(a_te.sum())

    print(f"features  : {args.features} ({len(cols)})")
    print(f"rows      : train {len(ytr):,} / val {len(yva):,} / "
          f"test {len(yte):,}  ({n_att_te:,} attacks in test)")
    print(f"ranking at: FPR <= {target_fpr:.4%}")
    print(f"cost model: prior {args.prior:g}, miss {args.cost_fn:g}x a false alarm")
    print(f"resolution: 0.001 recall = {0.001 * n_att_te:.0f} attacks on this "
          f"test set")

    # Bagged forests are ~10-20x slower than the boosters at 2M rows and hold
    # every tree in RAM. Say so before committing, and offer the staged run.
    slow = [m for m in want if m in ("RandomForest", "ExtraTrees")]
    if slow and len(ytr) > SLOW_WARN_ROWS:
        cheap = [m for m in want if m not in slow]
        print(f"\nNOTE: {', '.join(slow)} on {len(ytr):,} rows will dominate the "
              f"runtime and memory.")
        if cheap:
            print(f"      To stage it, run the cheap candidates first:\n"
                  f"        --models \"{','.join(cheap)}\"\n"
                  f"      then the forests separately:\n"
                  f"        --models \"{','.join(slow)}\"")
        else:
            print(f"      Only the forests were selected, so there is nothing to "
                  f"stage ahead of them.")
        print(f"      Either way the results MERGE into one report, so a staged "
              f"run\n      does not lose the earlier models.\n")
    print()

    rows = []
    flagged: dict[str, np.ndarray] = {}
    for name in want:
        est, needs_w = all_c[name]
        t0 = time.time()
        est.fit(Xtr, ytr, sample_weight=w_tr) if needs_w else est.fit(Xtr, ytr)
        s_va = 1.0 - est.predict_proba(Xva)[:, list(est.classes_).index("Benign")]
        s_te = 1.0 - est.predict_proba(Xte)[:, list(est.classes_).index("Benign")]
        rv = recall_at_fpr(s_va, a_va, target_fpr)
        rt = recall_at_fpr(s_te, a_te, target_fpr)
        dt = deployment(rt, args.prior, args.cost_fp, args.cost_fn)
        # keep the test decisions so models can be compared pairwise, not just
        # by their marginal recall
        flagged[name] = s_te >= rv["threshold"]

        # The threshold you would actually ship is the one chosen on VAL.
        # Applying it to test gives the honest deployment number; `rt` above is
        # the ROC point, which re-tunes on test and is therefore optimistic.
        fl = a_te & (yte == "Flood")
        rt_val = eval_at(s_te, a_te, rv["threshold"], fl)
        dt_val = deployment(rt_val, args.prior, args.cost_fp, args.cost_fn)
        fl_rec = rt_val["flood_recall"]

        rows.append({"model": name, "fit_seconds": round(time.time() - t0, 1),
                     "val": rv, "test": rt, "test_flood_recall": fl_rec,
                     "deployment": dt,
                     "test_at_val_threshold": rt_val,
                     "deployment_at_val_threshold": dt_val})
        print(f"  {name:14s} recall@FPR  val {rv['recall']:.4f} | "
              f"test {rt['recall']:.4f}  (FPR {rt['fpr']:.4%})  "
              f"Flood {fl_rec:.4f}  cost {dt['cost_per_1k']:.2f}/1k   "
              f"[{time.time()-t0:.0f}s]")

    rows.sort(key=lambda r: -r["test"]["recall"])
    print("\n" + "=" * 96)
    print(f"RANKING BY TEST RECALL AT FPR <= {target_fpr:.4%}   "
          f"(prior {args.prior:g}, miss {args.cost_fn:g}x FP)")
    print("=" * 96)
    print(f"{'model':>14}{'test recall':>13}{'val recall':>12}"
          f"{'Flood recall':>14}{'FP':>8}{'FN':>8}"
          f"{'miss/1k':>10}{'cost/1k':>10}{'fit s':>8}")
    best = rows[0]["test"]["recall"]
    for r in rows:
        gap = best - r["test"]["recall"]
        d = r["deployment"]
        tag = "  (ref)" if r["model"] in REFERENCE_MODELS else ""
        print(f"{r['model']:>14}{r['test']['recall']:>13.4f}"
              f"{r['val']['recall']:>12.4f}{r['test_flood_recall']:>14.4f}"
              f"{r['test']['false_positives']:>8,}"
              f"{r['test']['false_negatives']:>8,}"
              f"{d['misses_per_1k']:>10.4f}{d['cost_per_1k']:>10.2f}"
              f"{r['fit_seconds']:>8.1f}"
              + ("" if gap < 1e-9 else f"   -{gap:.4f}") + tag)

    # ---- what you would actually see in production ------------------------
    # The table above re-tunes the threshold on TEST (the ROC point). This one
    # applies the VAL-chosen threshold, which is what gets shipped.
    print("\n" + "=" * 96)
    print("SAME MODELS AT THE VAL-CHOSEN THRESHOLD  (what actually gets deployed)")
    print("=" * 96)
    print(f"{'model':>14}{'tau (val)':>12}{'test FPR':>11}{'test recall':>13}"
          f"{'Flood recall':>14}{'FP':>8}{'FN':>8}{'miss/1k':>10}{'cost/1k':>10}")
    for r in sorted(rows, key=lambda r: -r["test_at_val_threshold"]["recall"]):
        v, d = r["test_at_val_threshold"], r["deployment_at_val_threshold"]
        tag = "  (ref)" if r["model"] in REFERENCE_MODELS else ""
        print(f"{r['model']:>14}{v['threshold']:>12.5f}{v['fpr']:>11.4%}"
              f"{v['recall']:>13.4f}{v['flood_recall']:>14.4f}"
              f"{v['false_positives']:>8,}{v['false_negatives']:>8,}"
              f"{d['misses_per_1k']:>10.4f}{d['cost_per_1k']:>10.2f}{tag}")
    print("  (FPR drifts off the 0.2985% target here -- that drift is the honest")
    print("   cost of not being allowed to tune on test.)")

    # ---- verdict -----------------------------------------------------------
    # The old rule compared best against WORST. With LogReg in the set as a
    # linear reference point, "worst" is always LogReg by a mile, so the spread
    # was always huge and the script always said "materially ahead" -- even when
    # the real candidates were within a few dozen attacks of each other. Rank
    # against the INCUMBENT instead, and test the gap.
    real = [r for r in rows if r["model"] not in REFERENCE_MODELS]
    # The champion must be a real candidate. A reference model topping the table
    # means the metric or the data is off, not that you should ship LogReg.
    champ = real[0] if real else rows[0]
    d0 = champ["deployment"]
    print(f"\nbest real candidate: {champ['model']} at cost "
          f"{d0['cost_per_1k']:.2f} per 1,000 flows "
          f"({d0['misses_per_1k']:.4f} misses, {d0['alerts_per_1k']:.2f} alerts)")
    if real:
        sp = real[0]["test"]["recall"] - real[-1]["test"]["recall"]
        print(f"spread among the {len(real)} real candidates: {sp:.4f} recall "
              f"points = {sp * n_att_te:.0f} attacks")
    for r in rows:
        if r["model"] in REFERENCE_MODELS:
            print(f"({r['model']} is a linear reference point, not a candidate: "
                  f"it is meant to lose.)")

    verdict: dict = {}
    inc = next((r for r in rows if r["model"] == INCUMBENT), None)
    if not real:
        print("\nonly reference models were run in this batch -- nothing to "
              "recommend.")
    elif inc is None:
        print(f"\n'{champ['model']}' leads THIS BATCH, but the incumbent "
              f"'{INCUMBENT}' was not\n"
              f"run here, so no paired test is possible. Include "
              f"'{INCUMBENT}' in the same\n"
              f"batch to get a McNemar comparison against it.")
    elif inc["model"] == champ["model"]:
        print(f"\nnothing beats the incumbent '{INCUMBENT}' on this run.")
    else:
        b = int((flagged[champ["model"]] & ~flagged[inc["model"]] & a_te).sum())
        c = int((flagged[inc["model"]] & ~flagged[champ["model"]] & a_te).sum())
        p = mcnemar_exact(b, c)
        gap = champ["test"]["recall"] - inc["test"]["recall"]
        saving = inc["deployment"]["cost_per_1k"] - champ["deployment"]["cost_per_1k"]
        verdict = {"challenger": champ["model"], "incumbent": INCUMBENT,
                   "challenger_only_attacks": b, "incumbent_only_attacks": c,
                   "recall_gap": gap, "cost_saving_per_1k": saving,
                   "mcnemar_p": p}
        print(f"\n'{champ['model']}' vs incumbent '{INCUMBENT}':")
        print(f"  recall gap {gap:+.4f} = {gap * n_att_te:+.0f} attacks, "
              f"cost {saving:+.2f}/1k")
        print(f"  disagreements on {b + c} attacks: {b} caught only by "
              f"{champ['model']}, {c} only by {INCUMBENT}")
        print(f"  exact McNemar p = {p:.4f}")
        # Practical and statistical significance are different questions, and
        # they can disagree: 14 attacks out of 39,594 can be a real, repeatable
        # effect (p < 0.01) and still not be worth changing a deployed model for.
        if gap < 0.005 and p >= 0.01:
            print("  -> NOT a real improvement: too small to matter AND not")
            print("     statistically distinguishable. Keep the incumbent.")
        elif gap < 0.005:
            print(f"  -> statistically detectable (p={p:.4f}) but TOO SMALL TO")
            print(f"     MATTER: {gap * n_att_te:.0f} attacks and "
                  f"{saving:.2f} cost/1k. Not worth changing a deployed model.")
            print("     Keep the incumbent; it is cheaper to run.")
        elif p >= 0.01:
            print("  -> a big-looking gap that is NOT statistically supported.")
            print("     Do not switch on this evidence.")
        else:
            print(f"  -> a real improvement: {gap * n_att_te:.0f} more attacks")
            print(f"     caught, {saving:.2f} cost/1k cheaper, p={p:.4f}.")
            print(f"     '{champ['model']}' is worth adopting.")
    for r in rows:
        # `verdict` stays {} when no paired test was possible (incumbent absent
        # from this batch); store null rather than an empty object so the JSON
        # does not look like a verdict that came back blank.
        r["verdict_vs_incumbent"] = (verdict or None) \
            if r["model"] == champ["model"] else None

    out = os.path.join(args.outdir, "compare_report.json")
    report = {"features": args.features, "target_fpr": target_fpr,
              "prior": args.prior, "cost_fp": args.cost_fp,
              "cost_fn": args.cost_fn,
              "n_train": int(len(ytr)), "n_val": int(len(yva)),
              "n_test": int(len(yte)),
              "skipped": bad, "incumbent": INCUMBENT,
              "reference_models": sorted(REFERENCE_MODELS),
              "ranking": rows}
    out, how = write_report(out, report, args.tag)
    print(f"\n{how} {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
