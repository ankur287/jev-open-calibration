#!/usr/bin/env python3
"""exp02_derived — reproduce Experiment 2's derived cascade numbers.

The draft ("Jev Is API-Only. Its Best Idea Is Open.") reports several
Experiment 2 numbers that exp02_cascade.py never saves to
data/exp02_results.json and, in one case, cannot produce at all:

  * "Read it off the dial": mean confidence of the items acted on at the
    fixed t=0.90 policy — 0.989 for the temperature-scaled model
    (measured precision 0.972) and 0.949 for plain logreg (measured
    precision 0.995). exp02_cascade.py computes neither.
  * The unlabeled thresholds for a 0.95 mean-confidence target
    (t ~= 0.54 temp, measured precision 0.924; t ~= 0.90 plain, measured
    precision 0.995). exp02_cascade.py prints these when it runs but
    does not save them.
  * The isotonic detour: per-class isotonic calibration escalated 23.5%
    at the exact 95%-precision operating point. exp02_cascade.py has no
    isotonic backend at all — the number lives only in that file's
    header comment and the draft.

This script retrains the backends exactly as exp02_cascade.py does
(same TF-IDF features, same classifiers; the audit that motivated it
validated that the exact operating points reproduce the stored
exp02_results.json), adds the isotonic backend via
CalibratedClassifierCV(method="isotonic", cv=3) — the same construction
Experiment 1 uses — and saves every derived number above.

Usage: python code/exp02_derived.py
Writes data/exp02_derived.json and prints a summary.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
from calibration import TemperatureScaledLR  # noqa: E402
from exp02_cascade import (  # noqa: E402
    exact_operating_point,
    unlabeled_threshold_for_target,
)

FIXED_T = 0.90
TARGET_PRECISION = 0.95
UNLABELED_TARGET = 0.95


def train_all():
    """Same training recipe as exp02_cascade.train_models, plus isotonic."""
    tr = pd.read_csv(os.path.join(BASE, "data", "banking77_train.csv"))
    te = pd.read_csv(os.path.join(BASE, "data", "banking77_test.csv"))
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr["text"])
    Xte = vec.transform(te["text"])
    ytr = tr["category"].to_numpy()
    yte = te["category"].to_numpy()
    models = {
        "naive_bayes": MultinomialNB(),
        "logreg": LogisticRegression(max_iter=1000, C=4.0),
        "logreg_temp": TemperatureScaledLR(),
        "logreg_isotonic": CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, C=4.0),
            method="isotonic", cv=3),
    }
    out = {}
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        pred = clf.classes_[proba.argmax(axis=1)]
        out[name] = {"top_p": proba.max(axis=1),
                     "correct": (pred == yte).astype(int)}
    return out


def derived_for(top_p, correct):
    auto = top_p >= FIXED_T
    fixed = {
        "threshold": FIXED_T,
        "coverage": float(auto.mean()),
        "n_auto": int(auto.sum()),
        "precision_auto": float(correct[auto].mean()) if auto.sum() else None,
        "mean_conf_auto": float(top_p[auto].mean()) if auto.sum() else None,
    }
    t = unlabeled_threshold_for_target(top_p, UNLABELED_TARGET)
    if t is None:
        unlabeled = {"target_mean_conf": UNLABELED_TARGET,
                     "threshold": None,
                     "note": "mean confidence of acted-on never reaches "
                             "the target"}
    else:
        a2 = top_p >= t
        unlabeled = {"target_mean_conf": UNLABELED_TARGET,
                     "threshold": float(t),
                     "coverage": float(a2.mean()),
                     "mean_conf_auto": float(top_p[a2].mean()),
                     "measured_precision": float(correct[a2].mean())}
    op = exact_operating_point(top_p, correct, TARGET_PRECISION)
    return {"accuracy": float(correct.mean()),
            "fixed_threshold": fixed,
            "unlabeled_threshold": unlabeled,
            "exact_operating_point_95": op}


def main():
    models = train_all()
    results = {
        "method": "retrains exp02_cascade.py's backends with its exact "
                  "recipe, plus logreg_isotonic via "
                  "CalibratedClassifierCV(method='isotonic', cv=3); "
                  "operating points from exp02_cascade's "
                  "exact_operating_point / unlabeled_threshold_for_target",
        "draft_reports": {
            "mean_conf_auto_at_0.90": {"logreg_temp": 0.989,
                                       "logreg": 0.949},
            "precision_auto_at_0.90": {"logreg_temp": 0.972,
                                       "logreg": 0.995},
            "unlabeled_threshold": {"logreg_temp": 0.54,
                                    "logreg": 0.90},
            "unlabeled_measured_precision": {"logreg_temp": 0.924,
                                             "logreg": 0.995},
            "isotonic_escalated_at_95": 0.235,
        },
        "models": {name: derived_for(d["top_p"], d["correct"])
                   for name, d in models.items()},
    }
    out_path = os.path.join(BASE, "data", "exp02_derived.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
