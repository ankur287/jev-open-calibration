#!/usr/bin/env python3
"""exp01_derived — reproduce Experiment 1's prose-only mean-confidence numbers.

The draft ("Jev Is API-Only. Its Best Idea Is Open.") reports two
Experiment 1 numbers that exp01_calibration.py never saves to
data/exp01_results.json (which stores only accuracy, ECE and Brier):

  * Naive Bayes on the 77-intent task is "badly underconfident":
    mean top probability 0.25 against 0.80 accuracy.
  * Plain logistic regression on the same task: mean top probability
    0.64 vs. 0.89 accuracy.

This script retrains the four backends exactly as exp01_calibration.py
does (same TF-IDF features, same classifiers, same 7-intent slice
selection) and saves the mean top probability for every backend on
both tasks, alongside the accuracy, so both prose numbers — and the
mean confidence of the two calibrated backends, which the draft does
not quote — reproduce from the repo.

Usage: python code/exp01_derived.py
Writes data/exp01_derived.json and prints a summary.
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

N_INTENTS_SMALL = 7


def load(n_intents=None):
    tr = pd.read_csv(os.path.join(BASE, "data", "banking77_train.csv"))
    te = pd.read_csv(os.path.join(BASE, "data", "banking77_test.csv"))
    if n_intents:
        keep = tr["category"].value_counts().nlargest(n_intents).index
        tr = tr[tr["category"].isin(keep)]
        te = te[te["category"].isin(keep)]
    return tr, te


def run(task, n_intents):
    tr, te = load(n_intents)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr["text"])
    Xte = vec.transform(te["text"])
    ytr = tr["category"].to_numpy()
    yte = te["category"].to_numpy()

    lr = LogisticRegression(max_iter=1000, C=4.0)
    models = {
        "naive_bayes": MultinomialNB(),
        "logreg": lr,
        "logreg_isotonic": CalibratedClassifierCV(lr, method="isotonic", cv=3),
        "logreg_temp": TemperatureScaledLR(),
    }
    results = {}
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        pred = clf.classes_[proba.argmax(axis=1)]
        results[name] = {
            "acc": float((pred == yte).mean()),
            "mean_top_p": float(proba.max(axis=1).mean()),
        }
        print(f"{task:16s} {name:16s} acc={results[name]['acc']:.4f}  "
              f"mean_top_p={results[name]['mean_top_p']:.4f}")
    return results


if __name__ == "__main__":
    all_results = {
        "banking77_full": run("banking77_full", None),
        "banking7": run("banking7", N_INTENTS_SMALL),
    }
    with open(os.path.join(BASE, "data", "exp01_derived.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nresults -> data/exp01_derived.json")
