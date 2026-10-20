"""From-scratch metrics cross-checked against scikit-learn (tests only)."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from fraudlab import metrics as M


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(7)
    y = (rng.random(5000) < 0.02).astype(int)
    scores = np.clip(rng.normal(0.2 + 0.5 * y, 0.2), 0, 1)
    scores[:50] = 0.5  # force ties
    return y, scores


def test_counts_match_sklearn(toy):
    y, s = toy
    for t in [0.0, 0.3, 0.5, 0.71, 1.0]:
        c = M.counts_at(y, s, t)
        tn, fp, fn, tp = confusion_matrix(y, (s >= t).astype(int), labels=[0, 1]).ravel()
        assert (c.tp, c.fp, c.fn, c.tn) == (tp, fp, fn, tn)


def test_average_precision_matches_sklearn(toy):
    y, s = toy
    assert M.average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_roc_auc_matches_sklearn(toy):
    y, s = toy
    assert M.roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_sweep_agrees_with_pointwise_counts(toy):
    y, s = toy
    sw = M.sweep(y, s)
    for i in [0, 10, len(sw["thresholds"]) // 2, len(sw["thresholds"]) - 1]:
        c = M.counts_at(y, s, sw["thresholds"][i])
        assert (sw["tp"][i], sw["fp"][i], sw["fn"][i], sw["tn"][i]) == (c.tp, c.fp, c.fn, c.tn)


def test_cost_curve_agrees_with_cost_at(toy):
    y, s = toy
    loss = np.full(len(y), 100.0)
    cc = M.cost_curve(y, s, loss, review_cost=5.0)
    for i in [1, 25, len(cc["thresholds"]) - 1]:
        direct = M.cost_at(y, s, loss, cc["thresholds"][i], 5.0)
        assert cc["cost"][i] == pytest.approx(direct["total"])
    assert cc["cost"][0] == pytest.approx(100.0 * y.sum())  # flag nothing


def test_best_fbeta_is_global_optimum(toy):
    y, s = toy
    t, f = M.best_fbeta_threshold(y, s, beta=2.0)
    grid = [M.summary(M.counts_at(y, s, u))["fbeta"] for u in np.unique(s)]
    assert f == pytest.approx(max(grid))
    assert M.summary(M.counts_at(y, s, t))["fbeta"] == pytest.approx(f)


def test_wilson_interval_known_value():
    lo, hi = M.wilson_interval(58, 70)
    assert lo == pytest.approx(0.7234, abs=1e-3) and hi == pytest.approx(0.8992, abs=1e-3)


def test_precision_undefined_when_nothing_flagged():
    c = M.Counts(tp=0, fp=0, fn=5, tn=95)
    assert np.isnan(M.summary(c)["precision"]) and M.summary(c)["fbeta"] == 0.0


def test_fraud_loss_is_always_an_array():
    amounts = np.array([10.0, 0.0, 250.0])
    assert M.fraud_loss_per_sample(amounts, 25.0, False).tolist() == [25.0, 25.0, 25.0]
    assert M.fraud_loss_per_sample(amounts, 25.0, True).tolist() == [35.0, 25.0, 275.0]
