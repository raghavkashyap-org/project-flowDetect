"""
Fingerprint detector -- shows HOW MUCH of your accuracy comes from constant,
tool-specific values rather than attack behaviour.

For every (class, feature) pair this computes:
    modal share  : fraction of that class's rows sitting at the single most
                   common value of the feature
    benign share : fraction of BENIGN rows that share that same value
    separability : modal_share * (1 - benign_share)
A pair with high modal share and ~0 benign share is a SINGLE-FEATURE RULE:
"if feature == value then class". Such rules are tool fingerprints (TCP stack
defaults, fixed segment sizes, scripted timing) and they are why a model can
score 99.9% on this dataset and still fail on a real network.

Streaming, one feature column at a time -> low memory, exact counts.

Usage:
    python3 fingerprints.py --input data_out/filtered_clean.parquet \
                            --outdir reports/fingerprints
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import FEATURES, LABEL_COL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="reports/fingerprints")
    ap.add_argument("--const-share", type=float, default=0.99,
                    help="modal share at/above which a (class,feature) pair is "
                         "called near-constant (default 0.99)")
    ap.add_argument("--benign-rare", type=float, default=0.01,
                    help="benign share at/below which the value is 'rare in "
                         "benign' (default 0.01)")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    labels = pq.read_table(args.input, columns=[LABEL_COL]) \
               .to_pandas()[LABEL_COL].astype(str).to_numpy()
    classes, codes = np.unique(labels, return_inverse=True)
    n = len(codes)
    print(f"rows={n:,}  classes={list(classes)}")
    if "Benign" not in list(classes):
        print("[FATAL] no 'Benign' class; need it as the reference")
        return 2
    bi = int(np.flatnonzero(classes == "Benign")[0])
    benign_mask = codes == bi
    n_benign = int(benign_mask.sum())

    rows = []
    for f in FEATURES:
        v = pq.read_table(args.input, columns=[f]).to_pandas()[f] \
              .to_numpy(dtype=np.float64)
        vid, uniq = pd.factorize(v)          # value -> id, consistent globally
        nb = len(uniq)
        ben_cnt = np.bincount(vid[benign_mask], minlength=nb)

        for ci, cname in enumerate(classes):
            m = codes == ci
            tot = int(m.sum())
            if tot == 0:
                continue
            cnt = np.bincount(vid[m], minlength=nb)
            top = int(cnt.argmax())
            share = cnt[top] / tot
            modal = float(uniq[top])
            bshare = float(ben_cnt[top]) / n_benign
            rows.append({
                "class": cname,
                "feature": f,
                "modal_value": modal,
                "modal_share": share,
                "benign_share": bshare,
                "separability": share * (1.0 - bshare),
                "n_rows": tot,
            })
        print(f"  {f:<22} done  ({time.time()-t0:5.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "fingerprints.csv"), index=False)

    # ---- summary ---------------------------------------------------------
    attack = df[df["class"] != "Benign"]
    near_const = attack[attack["modal_share"] >= args.const_share]
    rules = attack[(attack["modal_share"] >= 0.90) &
                   (attack["benign_share"] <= args.benign_rare)]

    print("\n" + "=" * 78)
    print("NEAR-CONSTANT (class, feature) PAIRS "
          f"(modal share >= {args.const_share:.0%}, attacks only)")
    print("=" * 78)
    if near_const.empty:
        print("  none")
    for cname, g in near_const.groupby("class"):
        print(f"\n  {cname}  ({int(g['n_rows'].iloc[0]):,} rows) — "
              f"{len(g)} of {len(FEATURES)} features are effectively constant")
        for _, r in g.sort_values("modal_share", ascending=False).head(8).iterrows():
            print(f"     {r['feature']:<22} == {r['modal_value']:<14.6g} "
                  f"in {r['modal_share']:6.2%} of class   "
                  f"(same value in {r['benign_share']:6.3%} of benign)")

    print("\n" + "=" * 78)
    print("STRONGEST SINGLE-FEATURE RULES "
          "(modal share >= 90% of class AND <= "
          f"{args.benign_rare:.0%} of benign)")
    print("=" * 78)
    print(f"{'class':<24}{'feature':<22}{'value':>12}{'class%':>9}{'benign%':>9}")
    for _, r in rules.sort_values("separability", ascending=False) \
                     .head(args.top).iterrows():
        print(f"{r['class']:<24}{r['feature']:<22}{r['modal_value']:>12.6g}"
              f"{r['modal_share']:>9.2%}{r['benign_share']:>9.3%}")

    # how many rows are covered by at least one strong rule
    print("\n" + "=" * 78)
    print("HOW MANY FEATURES ARE 'DEAD' (near-constant across ALL classes)?")
    dead = [f for f in FEATURES
            if (df[df.feature == f]["modal_share"] >= 0.999).all()]
    print(f"  {dead if dead else 'none'}")

    summary = {
        "rows": n,
        "classes": list(map(str, classes)),
        "n_features": len(FEATURES),
        "near_constant_pairs": int(len(near_const)),
        "strong_single_feature_rules": int(len(rules)),
        "per_class_near_constant_features": {
            str(c): int((g["modal_share"] >= args.const_share).sum())
            for c, g in attack.groupby("class")},
        "top_rules": rules.sort_values("separability", ascending=False)
                          .head(args.top).to_dict(orient="records"),
    }
    with open(os.path.join(args.outdir, "fingerprint_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {args.outdir}/fingerprints.csv and fingerprint_summary.json"
          f"   ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
