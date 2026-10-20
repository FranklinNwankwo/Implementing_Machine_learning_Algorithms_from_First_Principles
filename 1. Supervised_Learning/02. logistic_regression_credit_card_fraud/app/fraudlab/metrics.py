"""
Evaluation metrics, written from scratch in NumPy (no sklearn at runtime).

Conventions
-----------
* Positive class = fraud (y == 1).
* A transaction is flagged when its score is >= the threshold, matching the
  notebook's `predict(X, threshold)`.
* Threshold sweeps are exact: every distinct score is a candidate threshold,
  so curves have no grid artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def alerts(self) -> int:
        return self.tp + self.fp


def counts_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> Counts:
    pred = scores >= threshold
    pos = y == 1
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    tn = int(np.sum(~pred & ~pos))
    return Counts(tp, fp, fn, tn)


def safe_div(a: float, b: float, empty: float = 0.0) -> float:
    return float(a) / float(b) if b else empty


def summary(c: Counts, beta: float = 2.0) -> dict:
    """Point metrics for one operating point.

    Precision is undefined when nothing is flagged; it is reported as NaN
    rather than silently set to 0 or 1.
    """
    precision = safe_div(c.tp, c.tp + c.fp, empty=float("nan"))
    recall = safe_div(c.tp, c.tp + c.fn)
    fpr = safe_div(c.fp, c.fp + c.tn)
    if np.isnan(precision) or precision + recall == 0:
        fbeta = 0.0
    else:
        fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    denom = np.sqrt(float(c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn))
    mcc = (c.tp * c.tn - c.fp * c.fn) / denom if denom else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta,
        "fpr": fpr,
        "specificity": 1 - fpr,
        "mcc": float(mcc),
        "alert_rate": safe_div(c.alerts, c.n),
    }


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


# ─────────────────────────────────────────────────────────────────────────────
# Exact threshold sweeps
# ─────────────────────────────────────────────────────────────────────────────

def sweep(y: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Cumulative counts at every distinct score, highest threshold first.

    Returns arrays aligned on `thresholds` (descending). At thresholds[i],
    every sample with score >= thresholds[i] is flagged. Optional `weights`
    (e.g. transaction amounts) give the weighted fraud value caught/missed.
    """
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    pos = (y[order] == 1).astype(np.int64)
    w = np.ones_like(s) if weights is None else weights[order]
    last_of_block = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp = np.cumsum(pos)[last_of_block]
    fp = np.cumsum(1 - pos)[last_of_block]
    caught_value = np.cumsum(w * pos)[last_of_block]
    P, N = int(pos.sum()), int(len(pos) - pos.sum())
    return {
        "thresholds": s[last_of_block],
        "tp": tp, "fp": fp, "fn": P - tp, "tn": N - fp,
        "caught_value": caught_value,
        "total_fraud_value": float(np.sum(w * pos)),
        "P": P, "N": N,
    }


def pr_curve(y: np.ndarray, scores: np.ndarray) -> dict:
    sw = sweep(y, scores)
    precision = sw["tp"] / (sw["tp"] + sw["fp"])
    recall = sw["tp"] / sw["P"]
    # Anchor at (recall 0, precision 1) so the curve starts at the top-left.
    return {
        "recall": np.r_[0.0, recall],
        "precision": np.r_[1.0, precision],
        "thresholds": np.r_[np.inf, sw["thresholds"]],
    }


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """AP = Σ (R_n − R_{n−1}) · P_n, the step-wise area under the PR curve."""
    pr = pr_curve(y, scores)
    return float(np.sum(np.diff(pr["recall"]) * pr["precision"][1:]))


def roc_curve(y: np.ndarray, scores: np.ndarray) -> dict:
    sw = sweep(y, scores)
    return {
        "fpr": np.r_[0.0, sw["fp"] / sw["N"]],
        "tpr": np.r_[0.0, sw["tp"] / sw["P"]],
    }


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Exact ROC-AUC via the trapezoidal rule on the exact ROC curve
    (equivalent to the Mann–Whitney probability with ties counted as 1/2)."""
    r = roc_curve(y, scores)
    return float(np.sum(np.diff(r["fpr"]) * (r["tpr"][1:] + r["tpr"][:-1]) / 2))


def best_fbeta_threshold(y: np.ndarray, scores: np.ndarray, beta: float = 2.0) -> tuple[float, float]:
    sw = sweep(y, scores)
    p = sw["tp"] / (sw["tp"] + sw["fp"])
    r = sw["tp"] / sw["P"]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(p + r > 0, (1 + beta**2) * p * r / (beta**2 * p + r), 0.0)
    i = int(np.argmax(f))
    return float(sw["thresholds"][i]), float(f[i])


# ─────────────────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────────────────

def fraud_loss_per_sample(amounts: np.ndarray, fixed_fee: float, include_amount: bool) -> np.ndarray:
    """Loss incurred if a transaction is fraud and is NOT flagged."""
    amounts = np.asarray(amounts, dtype=np.float64)
    return np.full_like(amounts, float(fixed_fee)) + (amounts if include_amount else 0.0)


def cost_curve(y, scores, fraud_loss, review_cost: float) -> dict:
    """Total cost at every distinct threshold, plus the "flag nothing" point.

        cost(τ) = review_cost × alerts(τ) + Σ_{missed frauds} fraud_loss_i
    """
    sw = sweep(y, scores, weights=fraud_loss)
    alerts = sw["tp"] + sw["fp"]
    missed = sw["total_fraud_value"] - sw["caught_value"]
    return {
        "thresholds": np.r_[np.inf, sw["thresholds"]],
        "cost": np.r_[sw["total_fraud_value"], review_cost * alerts + missed],
        "alerts": np.r_[0, alerts],
        "missed_loss": np.r_[sw["total_fraud_value"], missed],
        "review": np.r_[0.0, review_cost * alerts],
    }


def cost_at(y, scores, fraud_loss, threshold, review_cost) -> dict:
    pred = scores >= threshold
    pos = y == 1
    alerts = int(pred.sum())
    missed = float(fraud_loss[pos & ~pred].sum())
    return {"review": review_cost * alerts, "missed": missed, "total": review_cost * alerts + missed,
            "flag_nothing": float(fraud_loss[pos].sum())}


def min_cost_threshold(y, scores, fraud_loss, review_cost) -> float:
    """Threshold minimising total cost. Returns 1.0 + ε when flagging nothing is optimal."""
    cc = cost_curve(y, scores, fraud_loss, review_cost)
    i = int(np.argmin(cc["cost"]))
    return float(cc["thresholds"][i])


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def brier(y, scores) -> float:
    return float(np.mean((scores - y) ** 2))


def reliability(y, scores, edges=(0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)) -> dict:
    """Reliability curve on fixed probability bins (empty bins dropped).

    Fixed, tail-heavy edges are used instead of equal-count bins because
    almost all frauds sit in the top 1% of scores.
    """
    edges = np.asarray(edges, dtype=float)
    which = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, len(edges) - 2)
    mean_pred, frac_pos, count, lo, hi = [], [], [], [], []
    for b in range(len(edges) - 1):
        mask = which == b
        if mask.any():
            mean_pred.append(scores[mask].mean()); frac_pos.append(y[mask].mean())
            count.append(int(mask.sum())); lo.append(edges[b]); hi.append(edges[b + 1])
    return {"mean_pred": np.array(mean_pred), "frac_pos": np.array(frac_pos), "count": np.array(count),
            "lo": np.array(lo), "hi": np.array(hi)}
