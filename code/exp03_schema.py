"""Exp 03: schema failure modes in open decision models.

Two failure patterns documented against Jev, reproduced here on open models
to show they are properties of the *interface paradigm*, not of one vendor:

Part 1 — the none-of-the-above trap.
  A Choice question is asked over options that exclude the true answer
  (mirrors the PriorBench finding: 30 off-schema messages, no "none of
  these" option, model picked a listed category every time at >=0.99).
  Setup: 7-intent classifier, evaluated on states from the other 70 intents.
  Measure the confidence distribution in-schema vs out-of-schema, and the
  false auto-act rate at a fixed threshold, for both an isotonic-calibrated
  and a temperature-scaled backend. The temperature comparison tests whether
  better in-distribution calibration helps or hurts the trap. The fix under
  test: abstention via threshold (the schema must give the model a way to
  say "none fit").

Part 2 — framing inconsistency.
  The same underlying question asked two ways gives two answers (mirrors the
  documented Jev case: "is this a refund?" as Noul -> 0.22, as Choice ->
  0.01/0.99). Open-model analog: P(intent X) from a dedicated one-vs-rest
  binary classifier (Noul-style truth question) vs P(intent X) from a
  77-way multiclass model (Choice-style). Nothing reconciles the two.
  Measure correlation and the rate at which they contradict at matched
  act/abstain thresholds.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from calibration import TemperatureScaledLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "figures")
N_INTENTS = 7
ACT_T = 0.90


def load_all():
    tr = pd.read_csv(os.path.join(BASE, "data", "banking77_train.csv"))
    te = pd.read_csv(os.path.join(BASE, "data", "banking77_test.csv"))
    return tr, te


def top_intents(tr, n):
    return list(tr["category"].value_counts().nlargest(n).index)


def part1_none_of_the_above():
    tr, te = load_all()
    keep = top_intents(tr, N_INTENTS)
    tr7 = tr[tr["category"].isin(keep)]
    te7 = te[te["category"].isin(keep)]
    out = tr[~tr["category"].isin(keep)].sample(600, random_state=0)

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr7["text"])
    base = LogisticRegression(max_iter=1000, C=4.0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr, tr7["category"].to_numpy())

    pin = clf.predict_proba(vec.transform(te7["text"])).max(axis=1)
    pout = clf.predict_proba(vec.transform(out["text"])).max(axis=1)

    # Same trap, temperature-scaled backend. Note the comparison subtlety:
    # at a FIXED threshold the operating point moves (coverage rises too),
    # so the fair comparison is at MATCHED in-schema coverage.
    tclf = TemperatureScaledLR().fit(Xtr, tr7["category"].to_numpy())
    tpin = tclf.predict_proba(vec.transform(te7["text"])).max(axis=1)
    tpout = tclf.predict_proba(vec.transform(out["text"])).max(axis=1)

    def out_act_at_coverage(pin_, pout_, cov):
        t = float(np.quantile(pin_, 1 - cov))
        return t, float((pout_ >= t).mean())

    matched = []
    for cov in (0.50, 0.657, 0.80, 0.90, 0.921):
        ti, oi = out_act_at_coverage(pin, pout, cov)
        tt, ot = out_act_at_coverage(tpin, tpout, cov)
        matched.append({"in_coverage": cov,
                        "iso_threshold": ti, "iso_out_act": oi,
                        "temp_threshold": tt, "temp_out_act": ot})

    res = {
        "in_schema_n": int(len(pin)),
        "out_schema_n": int(len(pout)),
        "in_mean_top_p": float(pin.mean()),
        "out_mean_top_p": float(pout.mean()),
        "in_auto_act_rate": float((pin >= ACT_T).mean()),
        "out_false_auto_act_rate": float((pout >= ACT_T).mean()),
        "temp_in_mean_top_p": float(tpin.mean()),
        "temp_out_mean_top_p": float(tpout.mean()),
        "temp_in_auto_act_rate": float((tpin >= ACT_T).mean()),
        "temp_out_false_auto_act_rate": float((tpout >= ACT_T).mean()),
        "matched_in_coverage": matched,
    }
    print("\n--- Part 1: none-of-the-above trap ---")
    print(f"in-schema  (n={len(pin)}): mean top_p={pin.mean():.3f}, "
          f"auto-act@{ACT_T}={(pin >= ACT_T).mean()*100:.1f}%")
    print(f"out-schema (n={len(pout)}): mean top_p={pout.mean():.3f}, "
          f"FALSE auto-act@{ACT_T}={(pout >= ACT_T).mean()*100:.1f}%  <-- every one of these is wrong")
    print(f"[temp-scaled] in-schema: mean top_p={tpin.mean():.3f}, "
          f"auto-act@{ACT_T}={(tpin >= ACT_T).mean()*100:.1f}%")
    print(f"[temp-scaled] out-schema: mean top_p={tpout.mean():.3f}, "
          f"FALSE auto-act@{ACT_T}={(tpout >= ACT_T).mean()*100:.1f}%")
    print("matched in-coverage -> out-of-schema false-act rate (iso vs temp):")
    for m in matched:
        print(f"  in-cov {m['in_coverage']*100:5.1f}%: iso t={m['iso_threshold']:.3f} "
              f"out={m['iso_out_act']*100:5.1f}% | temp t={m['temp_threshold']:.3f} "
              f"out={m['temp_out_act']*100:5.1f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pin, bins=30, alpha=0.6, label="in-schema states", density=True)
    ax.hist(pout, bins=30, alpha=0.6, label="out-of-schema states (true answer not offered)", density=True)
    ax.axvline(ACT_T, color="red", ls="--", lw=1.5, label=f"act threshold {ACT_T}")
    ax.set_xlabel("top probability (model's confidence in its forced pick)")
    ax.set_ylabel("density")
    ax.set_title("Forced choice without 'none of the above':\nthe model still picks, and still sounds sure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "schema_none_of_the_above.png"), dpi=150)
    plt.close(fig)
    return res, (pin, pout)


def part2_framing():
    tr, te = load_all()
    target = top_intents(tr, 1)[0]
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr["text"])
    Xte = vec.transform(te["text"])
    ytr = tr["category"].to_numpy()

    # Noul-style: dedicated binary truth question "is this <target>?"
    bin_clf = LogisticRegression(max_iter=1000, C=4.0)
    bin_clf.fit(Xtr, (ytr == target).astype(int))
    p_noul = bin_clf.predict_proba(Xte)[:, 1]

    # Choice-style: 77-way multiclass, read off P(<target>)
    mc_clf = LogisticRegression(max_iter=1000, C=4.0)
    mc_clf.fit(Xtr, ytr)
    p_choice = mc_clf.predict_proba(Xte)[:, list(mc_clf.classes_).index(target)]

    diff = np.abs(p_noul - p_choice)
    act_n = p_noul >= ACT_T
    act_c = p_choice >= ACT_T
    res = {
        "target_intent": target,
        "n": int(len(te)),
        "mean_abs_diff": float(diff.mean()),
        "pearson_r": float(np.corrcoef(p_noul, p_choice)[0, 1]),
        "noul_act_rate": float(act_n.mean()),
        "choice_act_rate": float(act_c.mean()),
        "contradiction_rate": float((act_n != act_c).mean()),
        "noul_acts_choice_abstains": float((act_n & ~act_c).mean()),
        "choice_acts_noul_abstains": float((act_c & ~act_n).mean()),
    }
    print("\n--- Part 2: framing inconsistency ---")
    print(f"target intent: {target}")
    print(f"mean |P_noul - P_choice| = {diff.mean():.3f}, pearson r = {res['pearson_r']:.3f}")
    print(f"contradiction rate at act@{ACT_T}: {(act_n != act_c).mean()*100:.2f}% of states "
          f"(one framing says act, the other says abstain)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(p_choice, p_noul, s=4, alpha=0.25)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="agreement")
    ax.axvline(ACT_T, color="red", ls=":", lw=1)
    ax.axhline(ACT_T, color="red", ls=":", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"P({target}) as Choice (77-way)")
    ax.set_ylabel(f"P({target}) as Noul (binary truth question)")
    ax.set_title("Same question, two framings, two answers\n"
                 f"{res['contradiction_rate']*100:.1f}% contradict at act@{ACT_T}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "schema_framing.png"), dpi=150)
    plt.close(fig)
    return res


def main():
    os.makedirs(FIG, exist_ok=True)
    r1, _ = part1_none_of_the_above()
    r2 = part2_framing()
    with open(os.path.join(BASE, "data", "exp03_results.json"), "w") as f:
        json.dump({"part1_none_of_the_above": r1, "part2_framing": r2}, f, indent=2)
    print("\nresults -> data/exp03_results.json")


if __name__ == "__main__":
    main()
