"""ECE for the real Jev and the open models on the same 15-bin convention.

calibration.py defaults to 15 equal-width bins, which is what exp01 used
for the open models. The live Jev scripts (exp04-06) used 10 bins. This
script recomputes every Jev ECE from the raw responses with 15 bins, and
runs a paired bootstrap of (open - Jev) Choice ECE on the same 280 queries
of the 7-intent slice.

Usage: python code/ece_matched_bins.py
Writes data/ece_matched_bins.json and figures/reliability_jev_live.png.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import TemperatureScaledLR, plot_reliability_diagram  # noqa: E402
from calibration import expected_calibration_error as ece  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BINS, N_BOOT, SEED = 15, 10_000, 7


def load(name):
    with open(os.path.join(BASE, "data", name)) as f:
        return [json.loads(line) for line in f]


def ci(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def main():
    tr = pd.read_csv(os.path.join(BASE, "data", "banking77_train.csv"))
    te = pd.read_csv(os.path.join(BASE, "data", "banking77_test.csv"))
    keep = list(tr["category"].value_counts().nlargest(7).index)
    te7 = te[te["category"].isin(keep)].reset_index(drop=True)

    # Jev, 7-intent slice: Choice top label + all 7 Noul answers per query
    rows = sorted((r for r in load("exp04_raw.jsonl") if r["part"] == "A"), key=lambda r: r["i"])
    jy, jp, ny, npr = [], [], [], []
    for r in rows:
        ans, true = r["payload"]["answers"], te7.loc[r["i"], "category"]
        c = ans["intent"]
        jy.append(int(c["choice"] == true))
        jp.append(max(c["probabilities"].values()))
        for it in keep:
            ny.append(int(it == true))
            npr.append(ans[f"noul_{it}"]["noul"])
    jy, jp, ny, npr = map(np.array, (jy, jp, ny, npr))

    # Open temperature-scaled model on the same 280 queries
    tr7 = tr[tr["category"].isin(keep)]
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    m = TemperatureScaledLR().fit(vec.fit_transform(tr7["text"]), tr7["category"].to_numpy())
    P = m.predict_proba(vec.transform(te7["text"]))
    oy = (m.classes_[P.argmax(axis=1)] == te7["category"].to_numpy()).astype(int)
    op = P.max(axis=1)

    rng = np.random.default_rng(SEED)
    n, k = len(jy), len(keep)
    bj, bo, bd, bn = [], [], [], []
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        a, b = ece(jy[ix], jp[ix], N_BINS), ece(oy[ix], op[ix], N_BINS)
        bj.append(a); bo.append(b); bd.append(b - a)
        pix = (ix[:, None] * k + np.arange(k)).ravel()  # resample queries, keep their 7 pairs
        bn.append(ece(ny[pix], npr[pix], N_BINS))

    out = {"n_bins": N_BINS, "bootstrap": f"percentile, {N_BOOT} resamples, seed {SEED}",
           "jev_choice_7": {"acc": float(jy.mean()), "ece": ece(jy, jp, N_BINS), "ci95": ci(bj),
                            "share_above_0.99": float((jp > 0.99).mean())},
           "open_temp_7": {"acc": float(oy.mean()), "ece": ece(oy, op, N_BINS), "ci95": ci(bo)},
           "paired_diff_open_minus_jev": {"mean": float(np.mean(bd)), "ci95": ci(bd)},
           "jev_noul_pooled": {"ece": ece(ny, npr, N_BINS), "ci95_cluster_by_query": ci(bn),
                               "positives_mean_p": float(npr[ny == 1].mean()),
                               "positives_share_ge_0.9": float((npr[ny == 1] >= 0.9).mean())}}
    for tag, f in (("jev_77_terse", "exp05_raw.jsonl"), ("jev_77_threeshot", "exp06_raw.jsonl")):
        y, p = [], []
        for r in load(f):
            c = r["payload"]["answers"]["intent"]
            y.append(int(c["choice"] == te.loc[r["i"], "category"]))
            p.append(max(c["probabilities"].values()))
        out[tag] = {"acc": float(np.mean(y)), "ece": ece(np.array(y), np.array(p), N_BINS)}

    plot_reliability_diagram(jy, jp, n_bins=N_BINS,
                             title="Jev Choice answers, 280 banking queries",
                             path=os.path.join(BASE, "figures", "reliability_jev_live.png"))
    print(json.dumps(out, indent=1))
    with open(os.path.join(BASE, "data", "ece_matched_bins.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
