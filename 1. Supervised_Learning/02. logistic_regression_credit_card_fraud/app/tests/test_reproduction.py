"""The shipped artifacts must reproduce the notebook's reported operating points."""

import numpy as np
import pytest

from fraudlab import metrics as M
from fraudlab.model import load_eval_data, load_models, load_meta

NOTEBOOK = {"A": dict(tau=0.99971, tp=55, fn=15, fp=16), "B": dict(tau=0.79094, tp=57, fn=13, fp=28)}


@pytest.fixture(scope="module")
def setup():
    return load_models(), load_eval_data(), load_meta()


@pytest.mark.parametrize("key", ["A", "B"])
def test_operating_point_matches_notebook(setup, key):
    models, data, _ = setup
    m = models[key]
    c = M.counts_at(data.y_test, m.predict_proba(data.X_test), m.tau)
    exp = NOTEBOOK[key]
    assert m.tau == pytest.approx(exp["tau"], abs=5e-6)
    assert (c.tp, c.fn, c.fp) == (exp["tp"], exp["fn"], exp["fp"])


def test_split_sizes(setup):
    _, data, meta = setup
    assert len(data.y_test) == meta["n_test"] == 42557 and int(data.y_test.sum()) == 70
    assert len(data.y_val) == meta["n_val"] == 42559 and int(data.y_val.sum()) == 71


@pytest.mark.parametrize("key", ["A", "B"])
def test_contributions_reconstruct_probability(setup, key):
    models, data, _ = setup
    m = models[key]
    x = data.X_test[123]
    logit = m.bias + m.contributions(x).sum()
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(m.predict_proba(x.reshape(1, -1))[0], rel=1e-12)


def test_gradient_check_passed(setup):
    assert setup[2]["gradient_check_rel_error"] < 1e-5


@pytest.mark.parametrize("key", ["A", "B"])
def test_tau_is_exact_f2_optimum_on_validation(setup, key):
    """τ* must be exactly the F2-optimal validation score, bit for bit, so the app's
    validation counts and presets agree with the notebook."""
    models, data, _ = setup
    tau, _ = M.best_fbeta_threshold(data.y_val, models[key].predict_proba(data.X_val), beta=2.0)
    assert tau == models[key].tau
