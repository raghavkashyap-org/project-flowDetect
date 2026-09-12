#!/usr/bin/env python3
"""End-to-end check of the DEPLOYED artifact pair.

Nothing else in the pipeline proves that model.onnx + onnx_card.json, used the
way the service will use them, reproduces the metrics in model_card.json. This
does. It is the last thing to run before you call the model deployed.

It re-implements the service's inference path from scratch -- reading the
feature order, class order and tau from onnx_card.json rather than from the
Python model -- then scores the full test split and compares the operating-point
metrics against model_card.json.

If the feature order were wrong, or the benign column index were wrong, or the
gate were applied backwards, this fails loudly. Scoring code that imports the
joblib model would not catch any of those.

    python3 verify_deploy.py --deploy reports/deploy --splits data_out/splits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from common import LABEL_COL
from train_family import FAMILY


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default="reports/deploy")
    ap.add_argument("--splits", default="data_out/splits")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=50_000)
    ap.add_argument("--prior", type=float, default=0.001)
    ap.add_argument("--cost-fn", type=float, default=100.0)
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=2e-5,
                    help="allowed absolute difference on FPR/recall")
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        raise SystemExit("onnxruntime is not installed: pip install onnxruntime")

    failures: list[str] = []

    def check(ok: bool, msg: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            failures.append(msg)

    # ---------------------------------------------------------------- cards
    card_p = os.path.join(args.deploy, "onnx_card.json")
    mcard_p = os.path.join(args.deploy, "model_card.json")
    onnx_p = os.path.join(args.deploy, "model.onnx")
    for p in (card_p, mcard_p, onnx_p):
        if not os.path.exists(p):
            raise SystemExit(f"missing artifact: {p}")
    card = json.load(open(card_p))
    mcard = json.load(open(mcard_p))

    print("=" * 72)
    print("VERIFY DEPLOYED ARTIFACT")
    print("=" * 72)
    print(f"  onnx    : {onnx_p} ({os.path.getsize(onnx_p) / 1e6:.1f} MB)")
    print(f"  cards   : {card_p}, {mcard_p}\n")

    # --------------------------------------------------- artifact integrity
    # Checks the bytes on disk, not the loaded graph. A model that was
    # truncated in transfer, edited by hand, or swapped for a different
    # training run passes every numerical check below while being the wrong
    # artifact. This is the only check that catches that.
    print("--- artifact integrity ---")
    meta_p = os.path.join(args.deploy, "model_metadata.json")
    sha_p = os.path.join(args.deploy, "model.onnx.sha256")

    def sha256_of(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    actual = sha256_of(onnx_p)
    print(f"  model.onnx sha256 : {actual}")

    have_integrity = os.path.exists(sha_p) or os.path.exists(meta_p)
    check(have_integrity,
          "integrity files present (model.onnx.sha256 / model_metadata.json)"
          + ("" if have_integrity else
             " -- re-run export_onnx.py to generate them"))

    if os.path.exists(sha_p):
        want = open(sha_p).read().split()[0].strip().lower()
        check(want == actual,
              f"model.onnx matches model.onnx.sha256 ({want[:16]}...)")

    meta = None
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        a = meta.get("artifact", {})
        check(a.get("sha256", "").lower() == actual,
              f"model.onnx matches model_metadata.json ({str(a.get('sha256'))[:16]}...)")
        check(a.get("bytes") == os.path.getsize(onnx_p),
              f"byte size matches metadata ({a.get('bytes'):,})")
        # the metadata also pins the source joblib -- catch a graph exported
        # from a different training run than the card describes
        src = (meta.get("source") or {}).get("joblib_sha256")
        job = os.path.join(args.deploy, "model.joblib")
        if src and os.path.exists(job):
            check(sha256_of(job) == src,
                  "model.joblib matches the sha256 recorded at export time")
        # cross-check the two cards agree on the gate; they are written by
        # different scripts and can drift
        c = meta.get("contract", {})
        check(c.get("features") == card["input"]["features"],
              "metadata feature order matches onnx_card")
        check(abs(float(c.get("gate", {}).get("tau", 0))
                  - float(card["gate"]["tau"])) < 1e-12
              if c.get("gate") else False,
              "metadata tau matches onnx_card")
        check(c.get("benign_column_index") == card["gate"]["benign_column_index"],
              "metadata benign_column_index matches onnx_card")
        # Only the packages needed to RUN inference are required. Optional
        # converters (onnxmltools for LightGBM/XGBoost) are legitimately absent
        # on an HGB-only install -- failing on those would be a false alarm.
        env = meta.get("environment", {})
        ESSENTIAL = ("numpy", "sklearn", "onnx", "onnxruntime", "protobuf")
        absent = [k for k in ESSENTIAL if env.get(k) in ("absent", None)]
        check(not absent,
              "every package required for inference is still importable"
              + (f" (missing: {', '.join(absent)})" if absent else
                 f" ({', '.join(f'{k} {env[k]}' for k in ESSENTIAL)})"))
        opt = [f"{k} {v}" for k, v in env.items()
               if k not in ESSENTIAL and k not in ("python", "platform")]
        print(f"     optional/other: {', '.join(opt) if opt else 'none'}")
    print()

    cols = card["input"]["features"]
    in_name = card["input"]["name"]
    gate = card["gate"]
    tau = float(gate["tau"])
    bi = int(gate["benign_column_index"])
    classes = gate["classes"]

    print("--- contract from onnx_card.json ---")
    check(len(cols) == card["input"]["shape"][1] == mcard["n_features"],
          f"feature count agrees across cards ({len(cols)})")
    check(cols == mcard["features"],
          "feature ORDER in onnx_card matches model_card")
    check(abs(tau - float(mcard["threshold"])) < 1e-12,
          f"tau matches model_card ({tau:.6f})")
    check(classes[bi] == "Benign",
          f"benign_column_index {bi} really is Benign (classes are "
          f"{'alphabetical' if classes == sorted(classes) else 'NOT alphabetical'})")
    check(classes == sorted(classes),
          "class order is sklearn's sorted order -- read it from the card, "
          "never hardcode FAMILY_ORDER")

    # ---------------------------------------------------------------- data
    path = os.path.join(args.splits, f"{args.split}.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"missing split: {path}")
    need = cols + [LABEL_COL]
    df = pd.read_parquet(path, columns=need)
    X = df[cols].to_numpy(dtype=np.float32, copy=False)   # card order, float32
    y = np.array([FAMILY[v] for v in df[LABEL_COL]])
    print(f"\n--- data ---\n  {path}: {len(df):,} rows, "
          f"{(y != 'Benign').sum():,} attack / {(y == 'Benign').sum():,} benign")
    check(len(X) == mcard["test_metrics"]["n_attack"] + mcard["test_metrics"]["n_benign"],
          "row count matches what model_card.json was scored on")

    # ------------------------------------------------------------- inference
    sess = ort.InferenceSession(onnx_p, providers=["CPUExecutionProvider"])
    got_in = [i.name for i in sess.get_inputs()]
    got_out = [o.name for o in sess.get_outputs()]
    check(got_in == [in_name], f"graph input is {in_name} (got {got_in})")
    check("probabilities" in got_out, f"graph exposes probabilities (got {got_out})")

    probs = np.empty((len(X), len(classes)), dtype=np.float32)
    for s in range(0, len(X), args.batch):
        e = min(s + args.batch, len(X))
        probs[s:e] = sess.run(["probabilities"], {in_name: X[s:e]})[0]

    # the gate, exactly as the Alert Router must implement it
    score = 1.0 - probs[:, bi].astype(np.float64)
    alert = score >= tau

    is_att = y != "Benign"
    tp = int((alert & is_att).sum())
    fp = int((alert & ~is_att).sum())
    fn = int((~alert & is_att).sum())
    n_att, n_ben = int(is_att.sum()), int((~is_att).sum())
    fpr, rec = fp / n_ben, tp / n_att
    att_1k, ben_1k = 1000 * args.prior, 1000 * (1 - args.prior)
    cost = args.cost_fn * att_1k * (1 - rec) + args.cost_fp * ben_1k * fpr

    ref = mcard["test_metrics"]
    print("\n--- operating point: ONNX + gate  vs  model_card.json ---")
    print(f"  {'metric':16s}{'onnx+gate':>16s}{'model_card':>16s}{'delta':>14s}")
    for label, got, want, fmt in [
        ("TP", tp, ref["true_positives"], "{:,}"),
        ("FP", fp, ref["false_positives"], "{:,}"),
        ("FN", fn, ref["false_negatives"], "{:,}"),
        ("FPR", fpr, ref["fpr"], "{:.6%}"),
        ("recall", rec, ref["recall"], "{:.6%}"),
        ("precision", tp and (att_1k * rec) / (att_1k * rec + ben_1k * fpr),
         ref["precision"], "{:.6%}"),
        ("cost/1k", cost, ref["cost_per_1k"], "{:.4f}"),
    ]:
        d = abs(got - want)
        print(f"  {label:16s}{fmt.format(got):>16s}{fmt.format(want):>16s}"
              f"{d:>14.2e}")
        tol = args.tol if label in ("FPR", "recall", "precision") else \
            (0.01 if label == "cost/1k" else 0.5)
        check(d <= tol, f"{label} reproduces model_card within tolerance")

    fam_ok = True
    for fam, d in ref["per_family"].items():
        m = y == fam
        if fam == "Benign":
            got_n = int(alert[m].sum())
            fam_ok &= got_n == d["false_alarms"]
        else:
            got_n = int(alert[m].sum())
            fam_ok &= got_n == d["caught"]
    check(fam_ok, "per-family caught / false-alarm counts reproduce exactly")

    print("\n" + "=" * 72)
    if failures:
        print(f"FAILED -- {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED -- model.onnx + onnx_card.json reproduce")
    print("model_card.json on the full test split. Safe to wire into the service.")
    print(f"\nReminder: the gate lives OUTSIDE the graph. score = 1 - P(Benign)")
    print(f"using column {bi}; alert when score >= {tau:.6f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
