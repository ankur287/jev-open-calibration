#!/usr/bin/env python3
"""bootstrap_cis — reproduce the draft's bootstrap CIs for the real-Jev ECEs.

The draft ("Jev Is API-Only. Its Best Idea Is Open.") reports two bootstrap
intervals for Experiment 4 that no script in this repo generated:

  * Jev Choice ECE 0.010, 95% CI 0.006-0.027   (280 items)
  * Jev Noul pooled ECE 0.071, 95% CI 0.066-0.076  (1,960 item-intent pairs)

This script recomputes both from data/exp04_raw.jsonl, offline, with the
method disclosed here so the numbers are reproducible as staged:

  * ECE: calibration.py's expected_calibration_error, 10 bins — the house
    convention the draft and Figure 5 use ((lo, hi] bin edges). exp04's own
    ece_score uses [lo, hi) edges and gives Choice ECE 0.009; the Choice CI
    under that convention is reported too, as a sensitivity line.
  * Interval: percentile bootstrap, 10,000 resamples, seed 7.
  * Noul: pairs are resampled as the draft's pooled number implies, and a
    cluster bootstrap (resampling the 280 items, 7 pairs each) is reported
    alongside, because pairs within an item are not independent.

Usage: python code/bootstrap_cis.py
Writes data/bootstrap_cis.json and prints a summary.
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
from calibration import expected_calibration_error  # noqa: E402

INTENT_LIST = [
    "card_payment_fee_charged",
    "direct_debit_payment_not_recognised",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "wrong_amount_of_cash_received",
    "cash_withdrawal_charge",
    "transaction_charged_twice",
    "declined_cash_withdrawal",
]
N_BOOT = 10_000
SEED = 7


def ece_house_fast(y_true, y_prob, n_bins=10):
    """Vectorised twin of calibration.expected_calibration_error (lo, hi].

    Bin b holds probs in (edges[b], edges[b+1]]; prob 0 falls in bin 0,
    matching the reference implementation's first-bin special case.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    idx = np.clip(np.ceil(y_prob * n_bins).astype(int) - 1, 0, n_bins - 1)
    idx = np.where(y_prob == 0.0, 0, idx)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        n_b = m.sum()
        if n_b:
            ece += (n_b / len(y_true)) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def ece_exp04_fast(y_true, y_prob, n_bins=10):
    """exp04_jev_live.ece_score convention: [lo, hi) bins via int(p * n)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    idx = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        n_b = m.sum()
        if n_b:
            ece += (n_b / len(y_true)) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def percentile_ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_rows(y_true, y_prob, ece_fn, rng):
    """Resample rows (items or pairs) with replacement."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = rng.integers(0, n, n)
        out[k] = ece_fn(y_true[idx], y_prob[idx])
    return out


def load_part_a():
    import pandas as pd

    te = pd.read_csv(os.path.join(BASE, "data", "banking77_test.csv"))
    te = te[te["category"].isin(INTENT_LIST)].reset_index(drop=True)
    rows = []
    with open(os.path.join(BASE, "data", "exp04_raw.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            if rec["part"] == "A":
                rows.append(rec)
    rows.sort(key=lambda r: r["i"])

    choice_y, choice_p = [], []
    noul_y, noul_p = [], []  # grouped per item: 280 x 7
    for rec in rows:
        true = te.loc[rec["i"], "category"]
        ans = rec["payload"]["answers"]
        ch = ans["intent"]
        choice_y.append(1 if ch["choice"] == true else 0)
        choice_p.append(ch["probabilities"][ch["choice"]])
        item_y, item_p = [], []
        for intent in INTENT_LIST:
            item_p.append(ans[f"noul_{intent}"]["noul"])
            item_y.append(1 if intent == true else 0)
        noul_y.append(item_y)
        noul_p.append(item_p)
    return (np.array(choice_y), np.array(choice_p),
            np.array(noul_y), np.array(noul_p))


def main():
    choice_y, choice_p, noul_y, noul_p = load_part_a()

    # Sanity: the fast ECE must match the reference implementation exactly.
    assert abs(ece_house_fast(choice_y, choice_p)
               - expected_calibration_error(choice_y, choice_p, 10)) < 1e-12
    assert abs(ece_house_fast(noul_y.ravel(), noul_p.ravel())
               - expected_calibration_error(
                   noul_y.ravel(), noul_p.ravel(), 10)) < 1e-12

    rng = np.random.default_rng(SEED)
    choice_vals = bootstrap_rows(choice_y, choice_p, ece_house_fast, rng)
    choice_vals_alt = bootstrap_rows(choice_y, choice_p, ece_exp04_fast, rng)
    noul_vals = bootstrap_rows(noul_y.ravel(), noul_p.ravel(),
                               ece_house_fast, rng)
    # Cluster bootstrap: resample items, keeping each item's 7 pairs together.
    cluster_vals = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = rng.integers(0, len(noul_y), len(noul_y))
        cluster_vals[k] = ece_house_fast(noul_y[idx].ravel(),
                                         noul_p[idx].ravel())

    results = {
        "method": "percentile bootstrap, 10000 resamples, seed 7; "
                  "ECE = calibration.py, 10 bins, (lo, hi] edges",
        "choice_ece": {
            "point": ece_house_fast(choice_y, choice_p),
            "ci95": percentile_ci(choice_vals),
            "draft_reports": [0.006, 0.027],
            "ci95_exp04_bin_edges": percentile_ci(choice_vals_alt),
        },
        "noul_pooled_ece": {
            "point": ece_house_fast(noul_y.ravel(), noul_p.ravel()),
            "ci95_pairs_resampled": percentile_ci(noul_vals),
            "ci95_cluster_by_item": percentile_ci(cluster_vals),
            "draft_reports": [0.066, 0.076],
        },
    }
    out_path = os.path.join(BASE, "data", "bootstrap_cis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
