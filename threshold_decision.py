"""
Re-optimise an already-computed threshold sweep under a deployment prior.

Why this exists:
  tune_threshold.py saves the FULL sweep (every achievable threshold with its
  exact FPR / recall / precision) in threshold_report.json. That sweep is
  model output, so it never needs recomputing. What DOES change is the
  question you ask of it: how much does a false alarm cost you, and how much
  attack is actually in your traffic?

  The class balance inside CICIDS2017 (~9.8% attack) is not the class balance
  of a live network. Costing raw validation rows therefore prices false alarms
  far too cheaply and drags the optimum towards a permissive gate. This script
  re-weights the saved sweep to a prior you supply and reports, for each
  candidate operating point, the numbers an operator actually cares about:
  alerts and misses per 1,000 flows, and deployment precision.

  No refitting. Reads one JSON, prints tables.

Usage:
    python3 threshold_decision.py --report reports/threshold/threshold_report.json

    # one attack per 10,000 flows, a miss costs 10x a false alarm
    python3 threshold_decision.py --prior 0.0001 --cost-fn 10

    # compare several priors and cost ratios at once
    python3 threshold_decision.py --prior 0.001,0.0001,0.00001 --cost-fn 1,10,100

    # also price specific false-positive targets under the same prior
    python3 threshold_decision.py --prior 0.0001 --show-fpr 0.0001,0.001,0.01
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def parse_list(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if x.strip() != ""]


# --------------------------------------------------------------------------
def reweight(sw: dict, prior: float, cost_fp: float, cost_fn: float) -> dict:
    """Deployment-weighted cost arrays for a saved sweep.

    Weights are normalised so that one validation row stands for one
    deployment flow: an attack row counts prior / p_dataset of a flow's
    attack traffic, a benign row counts (1 - prior) / (1 - p_dataset).
    """
    fpr = np.asarray(sw["fpr"], dtype=float)
    rec = np.asarray(sw["recall"], dtype=float)
    thr = np.asarray(sw["threshold"], dtype=float)

    # per 1,000 deployment flows
    att_1k = 1000.0 * prior
    ben_1k = 1000.0 * (1.0 - prior)
    tp_1k = att_1k * rec
    fn_1k = att_1k * (1.0 - rec)
    fp_1k = ben_1k * fpr

    prec = np.where((tp_1k + fp_1k) > 0, tp_1k / np.maximum(tp_1k + fp_1k, 1e-12),
                    1.0)
    cost_1k = cost_fn * fn_1k + cost_fp * fp_1k
    return {"threshold": thr, "fpr": fpr, "recall": rec, "precision": prec,
            "alerts_1k": tp_1k + fp_1k, "misses_1k": fn_1k,
            "fp_1k": fp_1k, "cost_1k": cost_1k}


def fmt(x: float) -> str:
    """Compact but honest formatting across four orders of magnitude."""
    if x == 0:
        return "0"
    if x >= 100:
        return f"{x:,.0f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.4f}"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report",
                    default="reports/threshold/threshold_report.json")
    ap.add_argument("--sweep", default="",
                    help="path to threshold_sweep.npz (the exact sweep). "
                         "Defaults to the .npz sitting next to --report. If it "
                         "is missing, falls back to the downsampled curve in "
                         "the JSON and says so.")
    ap.add_argument("--prior", default="0.0001",
                    help="deployment attack rate(s), comma-separated "
                         "(default 0.0001 = 1 attack per 10,000 flows)")
    ap.add_argument("--cost-fp", default="1",
                    help="cost of one false alarm, comma-separated (default 1)")
    ap.add_argument("--cost-fn", default="10",
                    help="cost of one missed attack, comma-separated "
                         "(default 10)")
    ap.add_argument("--show-fpr", default="",
                    help="optional comma-separated FPR targets to price under "
                         "the same prior/costs, e.g. 0.0001,0.001,0.01")
    args = ap.parse_args()

    with open(args.report) as fh:
        rep = json.load(fh)
    n_val = rep["n_val"]
    # Recover the dataset's own attack rate from the saved untuned block:
    # benign_fpr = FP / n_benign, so n_benign falls straight out of it.
    unt = rep["untuned_argmax"]
    n_ben = unt["false_positives"] / unt["benign_fpr"]
    p_ds = (n_val - n_ben) / n_val

    # Prefer the exact sweep. The JSON curve is downsampled (~4,000 points),
    # so optimising it lands on a NEARBY threshold rather than the true
    # optimum -- good to a couple of significant figures, not exact.
    npz = args.sweep or os.path.join(
        os.path.dirname(os.path.abspath(args.report)), "threshold_sweep.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        sw = {k: z[k] for k in ("threshold", "fpr", "recall", "precision", "f1")}
        src = f"{npz} (EXACT, {len(sw['threshold']):,} thresholds)"
    else:
        sw = rep["curve_validation"]
        src = (f"downsampled JSON curve ({len(sw['threshold']):,} of "
               f"{rep.get('sweep_points_evaluated', '?')} thresholds) "
               f"-- APPROXIMATE; re-run tune_threshold.py for the .npz")

    print("=" * 78)
    print("THRESHOLD DECISION TABLE")
    print("=" * 78)
    print(f"report          : {args.report}")
    print(f"validation      : n={n_val:,}")
    print(f"sweep source    : {src}")
    print(f"dataset balance : {p_ds:.4%} attack  <-- NOT your deployment balance")
    print(f"model           : {rep['features']} ({rep['n_features']} features), "
          f"class_weight={rep['class_weight']}")

    print(f"\nUNTUNED argmax reference: FPR {unt['benign_fpr']:.4%} "
          f"({unt['false_positives']:,} FP), recall {unt['attack_recall']:.4%}")

    # ---- optimum under each (prior, cost_fp, cost_fn) combination ----------
    print("\n" + "=" * 78)
    print("COST-OPTIMAL THRESHOLD UNDER EACH ASSUMPTION")
    print("=" * 78)
    hdr = (f"{'prior':>10}{'fn:fp':>9}{'thresh':>10}{'FPR':>11}{'recall':>10}"
           f"{'prec':>9}{'alerts/1k':>11}{'miss/1k':>10}{'cost/1k':>10}")
    for prior in parse_list(args.prior):
        if not 0.0 < prior < 1.0:
            raise SystemExit(f"--prior {prior} must be in (0, 1)")
        for cost_fp in parse_list(args.cost_fp):
            for cost_fn in parse_list(args.cost_fn):
                w = reweight(sw, prior, cost_fp, cost_fn)
                i = int(w["cost_1k"].argmin())
                if cost_fp == parse_list(args.cost_fp)[0] and \
                        cost_fn == parse_list(args.cost_fn)[0]:
                    print(hdr)
                    print("-" * len(hdr))
                print(f"{prior:>10g}{cost_fn / cost_fp:>9g}"
                      f"{w['threshold'][i]:>10.5f}{w['fpr'][i]:>11.4%}"
                      f"{w['recall'][i]:>10.2%}{w['precision'][i]:>9.2%}"
                      f"{fmt(w['alerts_1k'][i]):>11}"
                      f"{fmt(w['misses_1k'][i]):>10}"
                      f"{fmt(w['cost_1k'][i]):>10}")

    # ---- price explicit FPR targets under the same assumptions -------------
    targets = parse_list(args.show_fpr)
    if targets:
        priors = parse_list(args.prior)
        prior = priors[0]
        print("\n" + "=" * 78)
        print(f"FIXED FALSE-POSITIVE TARGETS PRICED AT prior={prior:g}")
        print("=" * 78)
        fpr_arr = np.asarray(sw["fpr"], dtype=float)
        for cost_fp in parse_list(args.cost_fp):
            for cost_fn in parse_list(args.cost_fn):
                w = reweight(sw, prior, cost_fp, cost_fn)
                i_opt = int(w["cost_1k"].argmin())
                print(f"\n  cost_fn:cost_fp = {cost_fn:g}:{cost_fp:g}   "
                      f"(optimum is FPR {w['fpr'][i_opt]:.4%})")
                print(f"  {'target FPR':>12}{'thresh':>10}{'FPR':>11}"
                      f"{'recall':>10}{'prec':>9}{'alerts/1k':>11}"
                      f"{'miss/1k':>10}{'cost/1k':>10}{'vs opt':>10}")
                for t in targets:
                    ok = np.flatnonzero(fpr_arr <= t)
                    i = int(ok[-1]) if len(ok) else 0
                    ratio = w["cost_1k"][i] / max(w["cost_1k"][i_opt], 1e-12)
                    print(f"  {t:>12.4%}{w['threshold'][i]:>10.5f}"
                          f"{w['fpr'][i]:>11.4%}{w['recall'][i]:>10.2%}"
                          f"{w['precision'][i]:>9.2%}"
                          f"{fmt(w['alerts_1k'][i]):>11}"
                          f"{fmt(w['misses_1k'][i]):>10}"
                          f"{fmt(w['cost_1k'][i]):>10}{ratio:>9.1f}x")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
