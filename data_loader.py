"""
Tiny convenience loader: filtered_clean.parquet / train.parquet etc.
-> (X_float64, y_str) arrays ready for a scikit-learn / XGBoost pipeline.

The two Init*Win columns still contain the legitimate -1 sentinel. That is a
CATEGORY (UDP / one-way flows), not noise -- encode it as a categorical rather
than feeding the raw -1 into a distance-based model. A helper is provided.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pyarrow.parquet as pq

from common import FEATURES, LABEL_COL, SENTINEL_MINUS_ONE_COLS

SENT_FEATS = [c for c in FEATURES if c in SENTINEL_MINUS_ONE_COLS]


def load_matrix(parquet_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X float64 [n,32], y string [n]) in FEATURES column order."""
    pdf = pq.read_table(parquet_path).to_pandas()
    X = pdf[FEATURES].astype(np.float64).to_numpy()
    y = pdf[LABEL_COL].astype(str).to_numpy()
    return X, y


def add_sentinel_flags(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Turn the -1 sentinel in Init*Win into one-hot 'is_-1' flags.

    Returns a wider matrix and the ordered list of its column names:
        cols = FEATURES + [f"{c}_is_sentinel" for the 2 sentinel cols]
    The -1 values in the original 2 columns are set to NaN (the real window
    width is unknown there); the paired _is_sentinel=1 column records the fact.
    """
    cols = list(FEATURES)
    idx = {c: FEATURES.index(c) for c in SENT_FEATS}
    out = X.copy()
    flags = np.zeros((X.shape[0], len(SENT_FEATS)), dtype=np.float64)
    for j, c in enumerate(SENT_FEATS):
        mask = out[:, idx[c]] == -1.0
        out[mask, idx[c]] = np.nan
        flags[mask, j] = 1.0
        cols.append(f"{c}_is_sentinel")
    return np.hstack([out, flags]), cols


def load_splits(dirpath: str) -> dict:
    """Load train/val/test and return dict with X,y plus class info.

    Returns:
      {'train': (X,y), 'val': (X,y), 'test': (X,y),
       'label2idx': ..., 'class_weight_train': {...},
       'features': [...], 'report': <split_report.json dict>}
    """
    out = {}
    for key in ("train", "val", "test"):
        out[key] = load_matrix(os.path.join(dirpath, f"{key}.parquet"))
    rep_path = os.path.join(dirpath, "split_report.json")
    rep = None
    if os.path.exists(rep_path):
        with open(rep_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        out["class_weight_train"] = rep["train_class_weights"]
    out["report"] = rep
    return out
