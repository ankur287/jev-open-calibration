"""Exp 03b: the framing test on open models, using exp04's Jev protocol.

exp03 part 2 compared framings for one target intent over all 77 classes,
where both framings almost never act, so its 0.03% contradiction rate is
not comparable to the live Jev number. This rerun matches exp04 part A:
the 280 queries of the 7-intent slice, the Choice framing read as the
multiclass top probability, the Noul framing read as a one-vs-rest binary
classifier's P(true) for the *predicted* intent, contradiction = the two
disagree on act (>= 0.90). Also reports the rate at matched action rates.

Usage: python code/exp03b_framing_jev_protocol.py
Writes data/exp03b_results.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import TemperatureScaledLR  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACT_T = 0.90


def framing(p_choice, p_noul):
    act_c = p_choice >= ACT_T
    act_n = p_noul >= ACT_T
    # matched action rates: Noul threshold that acts on as many queries as Choice
    t_match = float(np.quantile(p_noul, 1 - act_c.mean()))
    act_nm = p_noul >= t_match
    return {"choice_act_rate": float(act_c.mean()),
            "noul_act_rate": float(act_n.mean()),
            "contradiction_rate": float((act_c != act_n).mean()),
            "contradictions": int((act_c != act_n).sum()),
            "noul_threshold_matched": t_match,
            "contradiction_rate_matched": float((act_c != act_nm).mean()),
            "contradictions_matched": int((act_c != act_nm).sum())}


def main():
    tr = pd.read_csv(os.path.join(BASE, "data", "banking77_train.csv"))
    te = pd.read_csv(os.path.join(BASE, "data", "banking77_test.csv"))
    keep = tr["category"].value_counts().nlargest(7).index
    tr7, te7 = tr[tr["category"].isin(keep)], te[te["category"].isin(keep)]
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr, Xte = vec.fit_transform(tr7["text"]), vec.transform(te7["text"])
    ytr = tr7["category"].to_numpy()

    noul = {c: LogisticRegression(max_iter=1000, C=4.0).fit(Xtr, (ytr == c).astype(int))
            for c in sorted(keep)}
    out = {"n": int(Xte.shape[0])}
    for name, clf in [("logreg", LogisticRegression(max_iter=1000, C=4.0)),
                      ("logreg_temp", TemperatureScaledLR())]:
        clf.fit(Xtr, ytr)
        P = clf.predict_proba(Xte)
        pred = clf.classes_[P.argmax(axis=1)]
        p_choice = P.max(axis=1)
        p_noul = np.array([noul[c].predict_proba(Xte[i])[0, 1] for i, c in enumerate(pred)])
        out[name] = framing(p_choice, p_noul)
        print(name, json.dumps(out[name]))

    with open(os.path.join(BASE, "data", "exp03b_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
