"""Headless UI test: drive every control in the app and assert nothing raises."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def at():
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert not app.exception, app.exception
    return app


def ok(app):
    assert not app.exception, app.exception


def test_initial_render(at):
    assert len(at.tabs) == 4
    assert at.session_state["tau_A"] == pytest.approx(0.99971, abs=5e-6)
    assert at.session_state["tau_B"] == pytest.approx(0.79094, abs=5e-6)


def test_presets_and_extremes(at):
    radio = next(r for r in at.radio if r.key == "thr_model")
    radio.set_value(radio.options[1]).run(); ok(at)
    for label in ["Best F2", "Min cost"]:
        next(b for b in at.button if b.label == label).click().run(); ok(at)
    for tau in (0.0, 1.0):
        at.slider(key="tau_B").set_value(tau).run(); ok(at)


def test_validation_split_and_costs(at):
    split = at.sidebar.radio[0]
    split.set_value(split.options[1]).run(); ok(at)
    at.toggle[0].set_value(False).run(); ok(at)
    at.number_input[0].set_value(500.0).run(); ok(at)
    at.number_input[1].set_value(0.0).run(); ok(at)
    # €500 per review, €0 per missed fraud: flagging nothing is cheapest, which no
    # slider position can express, so the preset must be disabled and explained.
    assert next(b for b in at.button if b.label == "Min cost").disabled
    assert any("flagging nothing is cheapest" in i.value for i in at.info)


def test_transaction_explorer_groups(at):
    for option in at.selectbox[0].options:
        at.selectbox[0].select(option).run(); ok(at)
