"""The shipped artifacts must reproduce the notebook's reported operating points."""

import numpy as np
import pytest

from fraudlab import metrics as M
from fraudlab.model import load_eval_data, load_models, load_meta

NOTEBOOK = {"A": dict(tau=0.987, tp=58, fn=12, fp=29), "B": dict(tau=0.767, tp=58, fn=12, fp=29)}


@pytest.fixture(scope="module")
def setup():
    return load_models(), load_eval_data(), load_meta()


@pytest.mark.parametrize("key", ["A", "B"])
def test_operating_point_matches_notebook(setup, key):
    models, data, _ = setup
    m = models[key]
    c = M.counts_at(data.y_test, m.predict_proba(data.X_test), m.tau)
    exp = NOTEBOOK[key]
    assert round(m.tau, 3) == exp["tau"]
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
