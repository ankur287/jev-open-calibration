"""Exp 01: calibration of open decision models on BANKING77.

Compares probabilistic classifiers behind the System One Choice
interface on (a) the full 77-intent task and (b) a 7-intent slice mirroring
the reduction in the existing TDS Jev comparison piece:
  - MultinomialNB      (famously miscalibrated)
  - LogisticRegression (usually decent out of the box)
  - Isotonic-calibrated LogisticRegression (CalibratedClassifierCV,
    per-class: can change predictions)
  - Temperature-scaled LogisticRegression (single T on a held-out split:
    cannot change predictions, zero accuracy tax by construction)

Reports accuracy, ECE, Brier score + reliability diagrams, and demonstrates
the typed Choice interface on sample states.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import expected_calibration_error, plot_reliability_diagram, TemperatureScaledLR
from system_one import ChoiceQuestion, SklearnBackend

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "figures")

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
    print(f"\n=== {task} ({yte.shape[0]} test, {len(np.unique(yte))} intents) ===")
    results = {}
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        pred = clf.classes_[proba.argmax(axis=1)]
        acc = float((pred == yte).mean())
        # top-probability calibration: P(correct) vs max proba
        top_p = proba.max(axis=1)
        correct = (pred == yte).astype(int)
        ece = expected_calibration_error(correct, top_p)
        brier = float(np.mean([(brier_score_loss(correct, top_p))]))
        results[name] = {"acc": acc, "ece": ece, "brier": brier}
        print(f"{name:16s} acc={acc:.4f}  ECE={ece:.4f}  brier={brier:.4f}")
        plot_reliability_diagram(
            correct, top_p, title=f"{task}: {name} (acc={acc:.3f})",
            path=os.path.join(FIG, f"reliability_{task}_{name}.png"))
    return results, vec, models["logreg_isotonic"], te


def demo_interface(vec, clf):
    backend = SklearnBackend(clf, vectorizer=vec)
    intents = list(clf.classes_)
    q = ChoiceQuestion(
        name="department",
        instructions="Which banking intent does this message express?",
        criteria={i: f"intent {i}" for i in intents[:10]})
    states = [
        "I am still waiting on my card, it has been two weeks",
        "Why was I charged twice this month?",
    ]
    print("\n--- System One interface demo (Choice) ---")
    for s in states:
        ans = backend.answer(s, [q])["department"]
        top3 = sorted(ans.probabilities.items(), key=lambda kv: -kv[1])[:3]
        print(f"state: {s!r}\n  choice={ans.choice} confidence={ans.confidence:.3f} "
              f"top3={[(k, round(v, 3)) for k, v in top3]}")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    all_results = {}
    all_results["banking77_full"], _, _, _ = run("banking77_full", None)
    all_results["banking7"], vec7, clf7, te7 = run("banking7", N_INTENTS_SMALL)
    demo_interface(vec7, clf7)
    import json
    with open(os.path.join(BASE, "data", "exp01_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nresults -> data/exp01_results.json, figures -> figures/reliability_*.png")
