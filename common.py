"""Shared configuration for the CICIDS extract / filter pipeline.

This file is the SINGLE source of truth for:
  * the labels you want to keep (row filter), and
  * the 32 Tier-1/Tier-2 feature columns you want to keep (column filter).

Change a name here and every script downstream picks it up automatically.
All label strings are matched VERBATIM (the raw file spells the labels with the
upstream typos / spacing, e.g. "DDOS attack-HOIC"). Do not retype them.
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# 1) Row filter  --  the only classes we keep
#    (exact strings as they appear in the file, from the audit label table)
# ---------------------------------------------------------------------------
KEEP_LABELS: set[str] = {
    "Benign",
    "DDOS attack-HOIC",          # DDoS attacks-HOIC
    "DDoS attacks-LOIC-HTTP",    # DDoS-LOIC-HTTP
    "DoS attacks-Hulk",          # DoS Hulk
    "Bot",
    "SSH-Bruteforce",
}

# Classes deliberately DROPPED, with the measured reason. Kept here so the
# decision is documented and reversible.
EXCLUDED_LABELS: dict[str, str] = {
    "FTP-BruteForce":
        "Only 53 DISTINCT rows among 193,354 (45.1% of the class is one row "
        "repeated 87,297x). Cannot be trained or evaluated; its folds would "
        "land in a single split. SSH-Bruteforce (94,048 distinct rows) covers "
        "the brute-force family. Re-add to KEEP_LABELS to include it.",
}

# ---------------------------------------------------------------------------
# 2) Column filter --  the 32 Tier-1 / Tier-2 features (your list, verbatim)
#    These are the ONLY numeric columns written to the cleaned matrix.
# ---------------------------------------------------------------------------
FEATURES: list[str] = [
    # flow timing
    "Flow Duration",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Flow IAT Std",
    # volume / direction
    "Tot Fwd Pkts",
    "TotLen Fwd Pkts",
    "Flow Byts/s",
    "Down/Up Ratio",
    # packet size profile
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Max",
    "Pkt Size Avg",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Mean",
    # forward header / segment / payload
    "Fwd Header Len",
    "Fwd Seg Size Min",
    "Fwd Act Data Pkts",
    # inter-arrival (forward + backward) -- robot/metronome detector
    "Fwd IAT Std",
    "Bwd IAT Mean",
    "Bwd IAT Tot",
    # TCP handshake / flag intent
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "Fwd PSH Flags",
    # slow-attack timing (idle vs active)
    "Idle Mean",
    "Active Mean",
    # TCP server / client window (keep the -1 sentinel as a category)
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
]
# The two Init*Win columns LEGITIMATELY hold -1 (sentinel for UDP / non-TCP /
# one-way flows). Every other feature must be >= 0 and finite.
SENTINEL_MINUS_ONE_COLS: set[str] = {"Init Fwd Win Byts", "Init Bwd Win Byts"}

# Column that holds the ground-truth class label.
LABEL_COL = "Label"

# ---------------------------------------------------------------------------
# 3) Behaviour columns deliberately EXCLUDED (leakage channels from the audit)
#    Never added to the training matrix -- reported separately only as an
#    ablation baseline if you want it.
# ---------------------------------------------------------------------------
LEAKAGE_COLS: list[str] = [
    "Dst Port", "Protocol", "Timestamp",
    "Day", "Hour", "Minute", "Second",
]


# ---------------------------------------------------------------------------
# Canonical HistGradientBoosting configuration.
#
# Every script that trains the production family model MUST build its estimator
# with **HGB_PARAMS. A gradient-boosted model's iteration count depends on its
# early-stopping settings, so two scripts using different values produce two
# DIFFERENT models -- and a threshold tuned by one is then applied to the other.
#
# This drifted once already: deploy_model.py and compare_models.py omitted
# early_stopping / validation_fraction / n_iter_no_change, inherited sklearn's
# defaults (n_iter_no_change=10 instead of 20), stopped at 45 iterations
# instead of 55, and flipped ~3% of alerts at the tuned threshold.
HGB_PARAMS = {
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
}


class _XGBFamily:
    """XGBoost's sklearn API needs integer labels 0..K-1 for multiclass.

    Lives here, not in deploy_model.py, so pickled artefacts have a stable
    import path -- a class defined in __main__ cannot be unpickled by another
    script. Exposes `classes_` in the sorted order XGBoost uses internally.
    """

    def __init__(self, **kwargs):
        self._kw = kwargs

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier
        self.classes_, y_int = np.unique(y, return_inverse=True)
        self.est_ = XGBClassifier(**self._kw)
        self.est_.fit(X, y_int, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.classes_[self.est_.predict(X)]

    def predict_proba(self, X):
        return self.est_.predict_proba(X)
