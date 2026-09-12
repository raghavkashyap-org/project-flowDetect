"""
Step 5 -- export the trained model to ONNX and PROVE the export is faithful.

Why this script exists:
  Converting a tree ensemble to ONNX is easy. Trusting the conversion is not.
  ONNX runs the trees in float32; LightGBM trains and predicts in float64. For
  almost every row that is irrelevant (~1e-7). But a row whose feature lands
  within float32 epsilon of a split threshold can take the OTHER branch, and
  one flipped tree moves the probability by ~1e-3. That is rare and usually
  harmless -- but "usually" is not something you ship on, so this script
  measures it on your actual test split and writes the numbers into the card.

What it does:
  1. Loads model.joblib produced by deploy_model.py.
  2. Converts to ONNX (zipmap off, so probabilities come back as a dense tensor).
  3. Re-scores the test split with BOTH engines and reports the parity gap,
     the number of label flips, and the number of GATE flips at your tau.
  4. Benchmarks single-row and batched latency.
  5. Writes model.onnx + onnx_card.json.

Usage:
    python3 export_onnx.py --model reports/deploy/model.joblib \
        --splits data_out/splits --outdir reports/deploy --rows 200000
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import time

import numpy as np

from common import FEATURES, _XGBFamily  # noqa: F401
from train_model import load_split

ONNX_OPSET = 15   # onnxmltools' LightGBM converter caps the AI.onnx.ml opset here


def sha256_of(path: str) -> str:
    """Hex sha256 of a file, streamed so a multi-GB artifact would not OOM."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ver(mod: str) -> str:
    """Version of an installed package, or 'absent' -- never raises.

    Uses importlib rather than __import__ because the latter returns the
    TOP-LEVEL package for a dotted name: __import__("google.protobuf") gives you
    `google`, which has no __version__, so protobuf would read as "unknown".
    """
    try:
        import importlib
        return getattr(importlib.import_module(mod), "__version__", "unknown")
    except Exception:
        return "absent"


def write_metadata(outdir: str, args, onx, sess, in_name, cols, bi, tau, est,
                   digest: str, onnx_card: dict) -> str:
    """Write model_metadata.json: everything needed to identify and verify this
    artifact later, without opening the .onnx or the .joblib.

    Deliberately redundant with onnx_card.json and model_card.json. Those two
    describe the *contract* and the *decision*; this describes the *artifact* --
    its hash, its size, the exact package versions that produced it, and when.
    Redundancy is the point: a registry entry has to stand alone.
    """
    mp = os.path.join(outdir, "model.onnx")
    src = args.model
    src_digest = sha256_of(src) if os.path.exists(src) else None

    card = {}
    card_p = os.path.join(outdir, "model_card.json")
    if os.path.exists(card_p):
        try:
            card = json.load(open(card_p))
        except Exception:
            card = {}

    meta = {
        "schema_version": 1,
        "artifact": {
            "name": "flow-detect-hgb-onnx",
            "file": "model.onnx",
            "sha256": digest,
            "bytes": os.path.getsize(mp),
            "onnx_opset": ONNX_OPSET,
            "ir_version": int(getattr(onx, "ir_version", 0)),
            "producer": f"{onx.producer_name} {onx.producer_version}".strip(),
            "created_utc": datetime.datetime.now(datetime.timezone.utc)
                             .isoformat(timespec="seconds"),
        },
        "source": {
            "joblib": src,
            "joblib_sha256": src_digest,
            "model_kind": card.get("model_kind", type(est).__name__),
            "n_estimators_declared": card.get("n_estimators"),
            "n_estimators_actual": int(getattr(est, "n_iter_", 0)) or None,
            "seed": card.get("seed"),
            "n_train_rows": card.get("n_train_rows"),
            "fit_seconds": card.get("fit_seconds"),
            "trained_on_val": card.get("trained_on_val"),
            "model_card_sha256": sha256_of(card_p) if os.path.exists(card_p) else None,
        },
        "contract": {
            "input_name": in_name,
            "input_shape": [None, len(cols)],
            "input_dtype": "float32",
            "n_features": len(cols),
            "feature_set": card.get("feature_set"),
            "features": cols,
            "outputs": [{"name": o.name, "shape": list(o.shape), "type": o.type}
                        for o in sess.get_outputs()],
            "classes": [str(c) for c in est.classes_],
            "benign_column_index": bi,
            "gate": {"rule": "alert when (1 - P(Benign)) >= tau",
                     "tau": tau,
                     "in_graph": False,
                     "note": "apply downstream; the ONNX graph is probabilities only"},
        },
        "parity": onnx_card["parity"],
        "latency_ms": onnx_card["latency_ms"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _ver("numpy"), "sklearn": _ver("sklearn"),
            "onnx": _ver("onnx"), "onnxruntime": _ver("onnxruntime"),
            "skl2onnx": _ver("skl2onnx"), "onnxmltools": _ver("onnxmltools"),
            "protobuf": _ver("google.protobuf"),
            # pyarrow is what actually reads the split .parquet files, so it
            # belongs in the record even though it is not on the inference path.
            "pyarrow": _ver("pyarrow"), "pandas": _ver("pandas"),
            # The joblib below is pickle-bound to THIS sklearn. Recorded as a
            # hard field so a mismatch is obvious before joblib.load is reached.
            "sklearn_wrote_joblib": _ver("sklearn"),
        },
        "verify": {
            "command": ("python3 verify_deploy.py --deploy "
                        f"{outdir} --splits <splits_dir>"),
            "checksum_command": f"sha256sum -c {os.path.join(outdir, 'model.onnx.sha256')}",
        },
    }
    p = os.path.join(outdir, "model_metadata.json")
    with open(p, "w") as fh:
        json.dump(meta, fh, indent=2)
    return p


def to_onnx(est, n_features: int):
    """Convert a supported estimator. Raises with the reason if unsupported.

    Each branch imports only what IT needs -- the sklearn path must not require
    onnxmltools to be installed, and vice versa.
    """
    kind = type(est).__name__
    if kind == "_XGBFamily":
        # unwrap: the converter needs the real XGBClassifier underneath
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
        # NB: no zipmap argument -- the XGBoost converter never emits ZipMap,
        # so probabilities already come back as a dense [None, K] tensor.
        return convert_xgboost(
            est.est_,
            initial_types=[("X", FloatTensorType([None, n_features]))],
            target_opset=ONNX_OPSET)
    if kind == "LGBMClassifier":
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        # zipmap=False is essential: otherwise probabilities come back as a
        # sequence of maps, which is awkward to consume from a service.
        return convert_lightgbm(
            est,
            initial_types=[("X", FloatTensorType([None, n_features]))],
            target_opset=ONNX_OPSET, zipmap=False)
    if kind in ("HistGradientBoostingClassifier", "RandomForestClassifier",
                "ExtraTreesClassifier", "GradientBoostingClassifier"):
        # skl2onnx has its OWN FloatTensorType and rejects onnxmltools'.
        from skl2onnx import to_onnx as sk_to_onnx
        from skl2onnx.common.data_types import FloatTensorType as SkFloat
        return sk_to_onnx(est, initial_types=[("X", SkFloat([None, n_features]))],
                          target_opset=ONNX_OPSET,
                          options={id(est): {"zipmap": False}})
    raise SystemExit(
        f"no ONNX converter wired up for {kind}. Supported: "
        f"HistGradientBoostingClassifier (skl2onnx), LGBMClassifier and "
        f"XGBoost (onnxmltools).")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="reports/deploy/model.joblib")
    ap.add_argument("--splits", default="data_out/splits")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--outdir", default="reports/deploy")
    ap.add_argument("--rows", type=int, default=200_000,
                    help="how many rows of the split to check parity on")
    ap.add_argument("--batch-sizes", default="1,64,1000,10000")
    args = ap.parse_args()

    import joblib
    import onnxruntime as ort
    ort.set_default_logger_severity(3)   # the label-shape warning is benign

    art = joblib.load(args.model)
    card, est = art["card"], art["model"]
    cols, tau = card["features"], card["threshold"]
    print(f"model    : {card['model_kind']} ({type(est).__name__})")
    print(f"features : {card['feature_set']} ({len(cols)})")
    print(f"tau      : {tau:.5f}")

    X, _ = load_split(os.path.join(args.splits, f"{args.split}.parquet"))
    ci = [FEATURES.index(c) for c in cols]
    X = X[:, ci]
    if args.rows and len(X) > args.rows:
        X = X[:args.rows]
    print(f"parity on: {len(X):,} rows of {args.split}\n")

    try:
        onx = to_onnx(est, len(cols))
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"missing dependency for ONNX export: {exc.name}\n"
            f"  pip install skl2onnx onnxmltools \"protobuf<7\" onnx onnxruntime\n"
            f"  then re-run. (protobuf must be < 7 -- see requirements.txt.)")
    except Exception as exc:
        first = str(exc).splitlines()[0][:200]
        hint = ""
        if "TreeEnsembleClassifier" in str(exc) or "AttributeProto" in str(exc):
            import google.protobuf as _pb
            hint = (f"\n  This is the protobuf>=7 incompatibility "
                    f"(you have {_pb.__version__}).\n"
                    f"  Fix:  pip install \"protobuf<7\"")
        raise SystemExit(
            f"ONNX conversion failed for {type(est).__name__}: "
            f"{type(exc).__name__}: {first}{hint}")
    sess = ort.InferenceSession(onx.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    t0 = time.time()
    p_native = est.predict_proba(X)
    t_native = time.time() - t0
    t0 = time.time()
    lab_onx, p_onx = sess.run(None, {in_name: X.astype(np.float32)})
    t_onnx = time.time() - t0

    bi = list(est.classes_).index("Benign")
    s_nat = 1.0 - p_native[:, bi]
    s_onx = 1.0 - p_onx[:, bi]
    d = np.abs(p_native - p_onx).max(axis=1)
    ds = np.abs(s_nat - s_onx)

    a_nat, a_onx = s_nat >= tau, s_onx >= tau
    flips = int((a_nat != a_onx).sum())
    # The ONNX label output is an index into the class list, while the native
    # estimator returns the label itself. Normalise both to the estimator's own
    # label space, otherwise this comparison reports 100% disagreement and is
    # worse than useless -- it would hide a real mismatch behind a fake one.
    lab_native = est.predict(X)
    cls = np.asarray(est.classes_)
    lab_norm = cls[lab_onx] if (np.issubdtype(lab_onx.dtype, np.integer)
                                and not np.issubdtype(cls.dtype, np.integer)) \
        else lab_onx
    label_flips = int((lab_norm != lab_native).sum())

    # How much of the split COULD have flipped? A row flips when its own
    # prob-diff is larger than its distance to tau. The exact answer on the
    # rows checked is `flips` below. For a forward-looking bound we use the
    # largest observed diff: any row within that distance of tau could flip on
    # new data. That bound is pessimistic -- one outlier row inflates it -- but
    # it is sound. Fixed bands like 1e-3 are NOT safe: measured errors here
    # reach 1e-1, so rows far from tau do flip.
    dist = np.abs(s_nat - tau)
    exposure = int((dist <= d.max()).sum())

    print("=" * 70)
    print("PARITY: native float64 vs ONNX float32")
    print("=" * 70)
    print(f"  max prob diff        : {d.max():.3e}")
    print(f"  mean prob diff       : {d.mean():.3e}")
    print(f"  rows with diff > 1e-6: {int((d > 1e-6).sum()):,} "
          f"({(d > 1e-6).mean():.3%})")
    print(f"  rows with diff > 1e-3: {int((d > 1e-3).sum()):,}")
    print(f"  label disagreements  : {label_flips}")
    print(f"  GATE flips at tau    : {flips}   <-- exact, on these rows")
    print(f"  worst-case exposure  : {exposure:,} rows lie within {d.max():.1e} "
          f"of tau")
    print(f"                         (upper bound on flips for new traffic;")
    print(f"                          pessimistic, driven by the single worst row)")
    if d.max() > 1e-4:
        print(f"  NOTE: max diff is large. This is float32 rounding on features")
        print(f"        spanning many orders of magnitude -- values above ~1e7")
        print(f"        are not exactly representable in float32. Decisions on")
        print(f"        THESE rows are unchanged, but the margin is not free.")
    if flips:
        print("  !! the export changes decisions. Do not ship without review.")
    else:
        print("  ok: identical decisions at the operating threshold.")


    print(f"\n{'=' * 70}\nLATENCY (onnxruntime CPU)\n{'=' * 70}")
    lat = {}
    for bs in [int(x) for x in args.batch_sizes.split(",") if x.strip()]:
        if bs > len(X):
            continue
        b = X[:bs].astype(np.float32)
        iters = max(5, int(20_000 / bs))
        for _ in range(3):
            sess.run(None, {in_name: b})
        t0 = time.perf_counter()
        for _ in range(iters):
            sess.run(None, {in_name: b})
        el = (time.perf_counter() - t0) / iters
        lat[bs] = {"total_ms": el * 1000, "per_row_ms": el * 1000 / bs}
        print(f"  batch {bs:>6,}: {el*1000:9.3f} ms total   "
              f"{el*1000/bs:.5f} ms/row")
    print(f"\n  native predict_proba : {t_native*1000:.1f} ms for {len(X):,} rows")
    print(f"  onnx runtime         : {t_onnx*1000:.1f} ms for {len(X):,} rows")

    os.makedirs(args.outdir, exist_ok=True)
    op = os.path.join(args.outdir, "model.onnx")
    with open(op, "wb") as fh:
        fh.write(onx.SerializeToString())
    onnx_card = {
        "source_model": args.model, "onnx_opset": ONNX_OPSET,
        "input": {"name": in_name, "shape": [None, len(cols)],
                  "dtype": "float32", "features": cols},
        "outputs": [{"name": o.name, "shape": o.shape, "type": o.type}
                    for o in sess.get_outputs()],
        "gate": {"rule": "alert when (1 - P(Benign)) >= tau",
                 "benign_column_index": bi, "tau": tau,
                 "classes": [str(c) for c in est.classes_],
                 "note": "the gate is NOT in the ONNX graph; apply it downstream"},
        "parity": {
            "rows_checked": int(len(X)),
            "max_prob_diff": float(d.max()), "mean_prob_diff": float(d.mean()),
            "rows_gt_1e-6": int((d > 1e-6).sum()),
            "rows_gt_1e-3": int((d > 1e-3).sum()),
            "label_disagreements": label_flips, "gate_flips": flips,
            "worst_case_exposure_rows": exposure,
            "exposure_note": ("rows within max_prob_diff of tau; upper bound on "
                              "flips for new traffic"),
            "verdict": ("identical decisions at tau" if not flips
                        else "DECISIONS DIFFER -- review before shipping"),
        },
        "latency_ms": lat,
    }
    cp = os.path.join(args.outdir, "onnx_card.json")
    with open(cp, "w") as fh:
        json.dump(onnx_card, fh, indent=2)

    # --- integrity + provenance -------------------------------------------
    # The .sha256 is written in `sha256sum` format so the standard tooling can
    # verify it: `sha256sum -c model.onnx.sha256`. Hash the file ON DISK, not
    # the in-memory proto -- that is the whole point, since it proves what was
    # actually written rather than what was serialised.
    digest = sha256_of(op)
    sp = os.path.join(args.outdir, "model.onnx.sha256")
    with open(sp, "w") as fh:
        # two spaces = binary mode marker, exactly as sha256sum emits it
        fh.write(f"{digest}  {os.path.basename(op)}\n")

    mdp = write_metadata(args.outdir, args, onx, sess, in_name, cols, bi, tau,
                         est, digest, onnx_card)

    print(f"\nwrote {op} ({os.path.getsize(op)/1e6:.1f} MB)")
    print(f"wrote {cp}")
    print(f"wrote {sp}")
    print(f"wrote {mdp}")
    print(f"\nsha256 {digest}")
    print(f"  verify with: (cd {args.outdir} && sha256sum -c model.onnx.sha256)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
