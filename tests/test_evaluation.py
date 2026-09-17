import numpy as np
import pytest

from src.evaluation import compare_models, diebold_mariano, directional_accuracy, mae, mape, rmse


def test_point_metrics():
    y, p = np.array([100.0, 200.0]), np.array([110.0, 190.0])
    assert rmse(y, p) == pytest.approx(10.0)
    assert mae(y, p) == pytest.approx(10.0)
    assert mape(y, p) == pytest.approx(7.5)


def test_directional_accuracy():
    current = np.array([100.0, 100.0, 100.0, 100.0])
    actual = np.array([101.0, 99.0, 101.0, 100.0])  # last day: no move, excluded
    pred = np.array([102.0, 98.0, 99.0, 150.0])
    assert directional_accuracy(current, actual, pred) == pytest.approx(200 / 3)
    assert np.isnan(directional_accuracy(current, actual, current))  # flat forecaster


def test_compare_models_sorted_by_rmse():
    y = np.array([1.0, 2.0, 3.0])
    table = compare_models(y, y + 0.5, {"bad": y + 5, "good": y + 0.1})
    assert list(table.index) == ["good", "bad"]


def test_diebold_mariano_detects_clearly_better_forecast():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    good = y + rng.normal(scale=0.1, size=500)
    bad = y + rng.normal(scale=1.0, size=500)
    res = diebold_mariano(y, good, bad)
    assert res["dm_stat"] < 0 and res["p_value"] < 0.01
    assert diebold_mariano(y, good, good)["p_value"] != diebold_mariano(y, good, good)["p_value"]  # NaN when identical
