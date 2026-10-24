"""Exp 02: confidence-gated cascade economics.

The System One pitch is that honest probabilities make abstention routable:
auto-act when confident, escalate when not. This experiment tests whether the
cascade actually works — and whether it depends on calibration being real.

Setup: BANKING77 full 77-intent task. Three backends behind the same Choice
interface:
  - naive_bayes      (weak + underconfident, ECE ~0.54)
  - logreg           (strong but underconfident, ECE ~0.25)
  - logreg_temp      (same predictions as logreg, temperature-scaled on a
                      held-out split: ECE ~0.02, zero accuracy tax)

Per-class isotonic calibration was tried and dropped from the cascade: its
per-class maps distorted cross-class ranking in the policy's operating region
(tied with logreg across the top ~30% of confidence-ranked items, worse below),
cost 1.7 points of accuracy, and escalated 23.5% vs 18.1% at the 95%-precision
operating point — so it does not isolate calibration from ranking.
Temperature scaling preserves the argmax exactly, which does.

Cascade policy: if top probability >= t, auto-decide; else escalate.
Escalated items are assumed resolved at FRONTIER_ACC (a frontier LLM / human
review tier) at COST_RATIO x the cheap tier. Both assumptions are stated, not
measured — the measured part is how precision and escalation trade off under
each model's own confidence scores.

Key question: at a fixed precision target on auto-decided items, how much
traffic must each model escalate? A miscalibrated model's confidence scores
should fail to separate right from wrong answers, forcing wasteful escalation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from calibration import TemperatureScaledLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "figures")

FRONTIER_ACC = 0.97   # assumed accuracy of the escalation tier (stated)
COST_RATIO = 100.0    # assumed cost multiple of escalation vs cheap tier (stated)
THRESHOLDS = np.round(np.arange(0.10, 1.00, 0.02), 2)


def load_data(n_intents=None):
    tr = pd.read_csv(os.path.join(BASE, "data", "banking77_train.csv"))
    te = pd.read_csv(os.path.join(BASE, "data", "banking77_test.csv"))
    if n_intents:
        keep = tr["category"].value_counts().nlargest(n_intents).index
        tr = tr[tr["category"].isin(keep)]
        te = te[te["category"].isin(keep)]
    return tr, te


def train_models(n_intents=None):
    tr, te = load_data(n_intents)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr["text"])
    Xte = vec.transform(te["text"])
    ytr = tr["category"].to_numpy()
    yte = te["category"].to_numpy()
    models = {
        "naive_bayes": MultinomialNB(),
        # same model before/after calibration: isolates calibration from ranking
        "logreg": LogisticRegression(max_iter=1000, C=4.0),
        # temperature scaling preserves argmax: accuracy identical to logreg
        "logreg_temp": TemperatureScaledLR(),
    }
    out = {}
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        pred = clf.classes_[proba.argmax(axis=1)]
        out[name] = {"top_p": proba.max(axis=1),
                     "correct": (pred == yte).astype(int)}
    return out


def cascade_curve(top_p, correct):
    rows = []
    for t in THRESHOLDS:
        auto = top_p >= t
        cov = float(auto.mean())
        prec = float(correct[auto].mean()) if auto.sum() else float("nan")
        esc = 1.0 - cov
        overall = cov * prec + esc * FRONTIER_ACC if auto.sum() else FRONTIER_ACC
        cost = cov * 1.0 + esc * COST_RATIO  # in cheap-tier units
        rows.append({"threshold": float(t), "coverage": cov,
                     "precision_auto": prec, "escalated": esc,
                     "overall_acc": float(overall), "rel_cost": float(cost)})
    return rows


def exact_operating_point(top_p, correct, target):
    """Grid-free operating point: over every distinct score used as a
    threshold, take the largest auto set with precision >= target.
    Sorting + unique thresholds removes grid artifacts entirely."""
    best = None
    for t in np.unique(top_p):
        auto = top_p >= t
        if not auto.sum():
            continue
        prec = float(correct[auto].mean())
        if prec >= target:
            esc = 1.0 - float(auto.mean())
            if best is None or esc < best["escalated"]:
                best = {"escalated": esc, "threshold": float(t),
                        "precision_auto": prec,
                        "overall_acc": float(auto.mean() * prec
                                            + (1.0 - auto.mean()) * FRONTIER_ACC),
                        "rel_cost": float(auto.mean() * 1.0
                                          + (1.0 - auto.mean()) * COST_RATIO)}
    return best


def unlabeled_threshold_for_target(top_p, target):
    """What you could compute without labels: the smallest t (largest auto
    set) whose mean confidence hits the target. With top-bin optimism this
    lands slightly short of the target in measured precision."""
    order = np.argsort(-top_p, kind="stable")
    s = top_p[order]
    cummean = np.cumsum(s) / np.arange(1, len(s) + 1)
    ok = np.where(cummean >= target)[0]
    if len(ok) == 0:
        return None
    k = ok[-1] + 1
    return float(s[k - 1])


def main():
    os.makedirs(FIG, exist_ok=True)
    # Full 77-intent task: base accuracy is 80-89%, so a 97% escalation tier
    # is a genuine upgrade and the cascade has real work to do.
    models = train_models(n_intents=None)
    print(f"\nTask: banking77_full, {sum(len(d['correct']) for d in models.values())//len(models)} test items")
    results = {name: cascade_curve(d["top_p"], d["correct"])
               for name, d in models.items()}

    print(f"\nAssumptions (stated, not measured): escalation tier acc={FRONTIER_ACC}, "
          f"cost ratio={COST_RATIO}x")
    print("\nOperating points (EXACT, grid-free): lowest escalation achieving precision target")
    exact = {}
    for target in (0.95, 0.98):
        print(f"  precision >= {target}:")
        exact[str(target)] = {}
        for name, d in models.items():
            op = exact_operating_point(d["top_p"], d["correct"], target)
            exact[str(target)][name] = op
            if op:
                print(f"    {name:16s} escalate {op['escalated']*100:5.1f}%  "
                      f"(t={op['threshold']:.4f})  overall acc={op['overall_acc']:.4f}  "
                      f"rel cost={op['rel_cost']:.1f}x")
            else:
                print(f"    {name:16s} target unreachable")

    print("\nUnlabeled threshold check: t where mean confidence of acted-on hits 0.95,")
    print("and the measured precision there (optimism => lands short of 0.95):")
    for name, d in models.items():
        t = unlabeled_threshold_for_target(d["top_p"], 0.95)
        if t is None:
            print(f"    {name:16s} mean confidence of acted-on never reaches 0.95 "
                  f"(model never confident enough for an unlabeled 0.95 policy)")
            continue
        auto = d["top_p"] >= t
        print(f"    {name:16s} t={t:.4f}  mean_conf={d['top_p'][auto].mean():.4f}  "
              f"measured_prec={d['correct'][auto].mean():.4f}  auto={auto.mean()*100:.1f}%")

    print("\nOperating points (grid, for the figure): lowest escalation achieving precision target on auto-decided")
    for target in (0.95, 0.98):
        print(f"  precision >= {target}:")
        for name, rows in results.items():
            feas = [r for r in rows if r["precision_auto"] >= target]
            if feas:
                best = min(feas, key=lambda r: r["escalated"])
                print(f"    {name:16s} escalate {best['escalated']*100:5.1f}%  "
                      f"(t={best['threshold']:.2f})  overall acc={best['overall_acc']:.4f}  "
                      f"rel cost={best['rel_cost']:.1f}x")
            else:
                print(f"    {name:16s} target unreachable")

    print("\nFixed-threshold policy (act iff top_p >= t):")
    fixed = {}
    for t in (0.80, 0.90, 0.98):
        print(f"  t={t:.2f}:")
        fixed[str(t)] = {}
        for name, rows in results.items():
            r = min(rows, key=lambda r: abs(r["threshold"] - t))
            fixed[str(t)][name] = r
            print(f"    {name:16s} auto {r['coverage']*100:5.1f}% @ prec={r['precision_auto']:.4f}  "
                  f"escalate {r['escalated']*100:5.1f}%  rel cost={r['rel_cost']:.1f}x")

    with open(os.path.join(BASE, "data", "exp02_results.json"), "w") as f:
        json.dump({"assumptions": {"frontier_acc": FRONTIER_ACC,
                                   "cost_ratio": COST_RATIO},
                   "curves": results,
                   "exact_operating_points": exact,
                   "fixed_thresholds": fixed}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, rows in results.items():
        esc = [r["escalated"] * 100 for r in rows]
        overall = [r["overall_acc"] * 100 for r in rows]
        ax.plot(esc, overall, marker="o", ms=3, label=name)
    ax.set_xlabel("% of traffic escalated to the expensive tier")
    ax.set_ylabel("overall accuracy (%)")
    ax.set_title("Cascade economics: temperature scaling preserves the tradeoff curve -\n"
                 "same predictions, honest thresholds; the weak model never gets there\n"
                 f"(escalation tier assumed {FRONTIER_ACC*100:.0f}% acc, {COST_RATIO:.0f}x cost)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG, "cascade_tradeoff.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nresults -> data/exp02_results.json, figure -> {path}")


if __name__ == "__main__":
    main()
