"""The lost-wallet example from the article's opening.

A query whose true intent (lost_or_stolen_card) is not among the 7 options
of the 7-intent slice. Every backend is forced to pick one of the seven.
This script fits the same 7-intent backends used in Experiments 1 and 3,
asks each one about the query through SklearnBackend, and routes the
answer with and without a "none of these" branch.

Usage: python code/example_lost_wallet.py   (a few seconds)
"""

import os
import sys

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import TemperatureScaledLR  # noqa: E402
from system_one import ChoiceQuestion, SklearnBackend  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERY = ("I lost my wallet today with all my credit cards. "
         "Will you please block the card and send a replacement?")
ACT_T = 0.90
MATCHED_T = 0.9989  # temperature model's threshold at 65.7% in-schema coverage (exp03)


def route(top_p, threshold, none_branch):
    if top_p >= threshold:
        return "ACT"
    return "NONE_OF_THESE" if none_branch else "ABSTAIN"


def main():
    tr = pd.read_csv(os.path.join(BASE, "data", "banking77_train.csv"))
    keep = tr["category"].value_counts().nlargest(7).index
    tr7 = tr[tr["category"].isin(keep)]

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X = vec.fit_transform(tr7["text"])
    y = tr7["category"].to_numpy()

    lr = lambda: LogisticRegression(max_iter=1000, C=4.0)  # noqa: E731
    backends = {
        "naive_bayes": MultinomialNB(),
        "logreg": lr(),
        "logreg_isotonic": CalibratedClassifierCV(lr(), method="isotonic", cv=3),
        "logreg_temp": TemperatureScaledLR(),
    }

    options = sorted(keep)
    print(f"query: {QUERY}")
    print(f"true intent: lost_or_stolen_card (in schema: {'lost_or_stolen_card' in options})\n")
    q = ChoiceQuestion(name="intent", instructions="Classify the query.",
                       criteria={c: "" for c in options})

    print(f"{'backend':18s} {'choice':28s} {'top_p':>6s}  at t={ACT_T}")
    for name, clf in backends.items():
        clf.fit(X, y)
        ans = SklearnBackend(clf, vectorizer=vec).answer(QUERY, [q])["intent"]
        top_p = ans.probabilities[ans.choice]
        print(f"{name:18s} {ans.choice:28s} {top_p:6.3f}  {route(top_p, ACT_T, False)}")
        if name == "logreg_temp":
            temp_top_p = top_p

    print(f"\nlogreg_temp at t={MATCHED_T} with a 'none of these' branch: "
          f"{route(temp_top_p, MATCHED_T, True)}")


if __name__ == "__main__":
    main()
