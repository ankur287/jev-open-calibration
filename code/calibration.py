"""Calibration metrics for the Jev article experiments.

Implements the standard evaluation toolkit for "honest probabilities":
Expected Calibration Error (ECE), reliability-diagram bin statistics, and
Brier score. Used to measure both the open-model reconstruction and the real
Jev API's reported probabilities.
"""

import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """ECE with equal-width bins: sum_b |acc(b) - conf(b)| * |b| / n."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = y_true[mask].mean()
        conf_b = y_prob[mask].mean()
        ece += (n_b / len(y_true)) * abs(acc_b - conf_b)
    return float(ece)


def reliability_diagram_data(y_true, y_prob, n_bins=15):
    """Per-bin (confidence, accuracy, count) for reliability diagrams."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        n_b = int(mask.sum())
        if n_b == 0:
            out.append({"bin_center": (lo + hi) / 2, "confidence": np.nan,
                        "accuracy": np.nan, "count": 0, "gap": np.nan})
            continue
        conf_b = float(y_prob[mask].mean())
        acc_b = float(y_true[mask].mean())
        out.append({"bin_center": (lo + hi) / 2, "confidence": conf_b,
                    "accuracy": acc_b, "count": n_b, "gap": acc_b - conf_b})
    return out


def plot_reliability_diagram(y_true, y_prob, n_bins=15, title="", path=None):
    """Reliability diagram + confidence histogram. Saves to path if given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = reliability_diagram_data(y_true, y_prob, n_bins)
    centers = np.array([s["bin_center"] for s in stats])
    conf = np.array([s["confidence"] for s in stats])
    acc = np.array([s["accuracy"] for s in stats])
    counts = np.array([s["count"] for s in stats])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6),
                                   gridspec_kw={"height_ratios": [3, 1]})
    valid = ~np.isnan(acc)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax1.bar(centers[valid], acc[valid], width=1.0 / n_bins * 0.9,
            alpha=0.7, edgecolor="black", label="observed accuracy")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("mean predicted probability (confidence)")
    ax1.set_ylabel("observed accuracy")
    ax1.set_title(title or "Reliability diagram")
    ax1.legend(loc="lower right")
    ax1.text(0.02, 0.95, f"ECE = {expected_calibration_error(y_true, y_prob, n_bins):.4f}",
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    ax2.bar(centers, counts, width=1.0 / n_bins * 0.9, color="gray", edgecolor="black")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("predicted probability")
    ax2.set_ylabel("count")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


class TemperatureScaledLR:
    """LogisticRegression + a single temperature fit on out-of-fold logits.

    Dividing all logits by one T > 0 cannot change the argmax, so accuracy
    is identical to the uncalibrated model by construction -- the accuracy
    tax is zero. Per-class methods (isotonic/sigmoid via
    CalibratedClassifierCV) lack this guarantee: their per-class maps can
    re-rank classes and change predictions. The temperature is fit on
    cross-fitted out-of-fold logits, so no training data is withheld from
    the base model.
    """

    def __init__(self, val_size=0.2, random_state=7, **lr_kwargs):
        self.val_size = val_size
        self.random_state = random_state
        self.lr_kwargs = lr_kwargs
        self.temperature_ = None
        self.classes_ = None

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from scipy.optimize import minimize_scalar

        kwargs = {"max_iter": 1000, "C": 4.0}
        kwargs.update(self.lr_kwargs)
        y = np.asarray(y)
        self._lr = LogisticRegression(**kwargs).fit(X, y)
        self.classes_ = self._lr.classes_
        # Out-of-fold logits, so T is fit on unseen data without
        # withholding training data from the base model.
        oof = np.zeros((len(y), len(self.classes_)))
        skf = StratifiedKFold(n_splits=3, shuffle=True,
                              random_state=self.random_state)
        for tri, cai in skf.split(X, y):
            m = LogisticRegression(**kwargs).fit(X[tri], y[tri])
            oof[cai] = m.decision_function(X[cai])
        cidx = np.where(self.classes_[None, :] == y[:, None])[1]

        def nll(T):
            z = oof / T
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)
            return -np.log(p[np.arange(len(y)), cidx] + 1e-12).mean()

        self.temperature_ = float(
            minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
        return self

    def predict_proba(self, X):
        z = self._lr.decision_function(X) / self.temperature_
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
