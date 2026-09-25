import json
import numpy as np
import pytest
from train import choose_threshold, evaluate, select_model, bootstrap_ci


def test_select_model_prefers_simpler_when_tied():
    scores = {"logistic_regression": 0.990, "random_forest": 0.9925, "neural_network": 0.9950}
    assert select_model(scores) == "logistic_regression"      # all within 0.005 of the best -> simplest wins


def test_select_model_takes_clear_winner():
    scores = {"logistic_regression": 0.90, "random_forest": 0.95, "neural_network": 0.94}
    assert select_model(scores) == "random_forest"


def test_choose_threshold_catches_all_malignant_when_asked():
    y = [0, 0, 0, 1, 1]
    p_mal = [0.9, 0.6, 0.3, 0.2, 0.1]
    t = choose_threshold(y, p_mal, target_recall=1.0)
    assert t == pytest.approx(0.3)
    assert evaluate(y, p_mal, t)["malignant_recall"] == 1.0


def test_choose_threshold_never_stricter_than_default():
    y, p_mal = [0, 0, 1, 1], [0.99, 0.98, 0.02, 0.01]
    assert choose_threshold(y, p_mal, 0.98) <= 0.5


def test_evaluate_confusion_counts():
    m = evaluate([0, 0, 1, 1], [0.9, 0.2, 0.1, 0.8], 0.5)["confusion"]
    assert m == {"tp_malignant_caught": 1, "fn_malignant_missed": 1, "fp_false_alarm": 1, "tn_benign_ok": 1}


def test_bootstrap_ci_is_ordered_and_bounded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(np.where(y == 0, 0.8, 0.2) + rng.normal(0, 0.2, 200), 0, 1)
    lo, hi = bootstrap_ci(y, p, 0.5, n=200)["accuracy"]
    assert 0 <= lo <= hi <= 1


def test_training_is_accurate_and_writes_files(trained_dir):
    m = json.load(open(trained_dir / "results" / "metrics.json"))
    assert m["test_at_default_threshold_0.5"]["accuracy"] > 0.93
    assert m["test_roc_auc"] > 0.97
    for f in ("models/model.joblib", "models/meta.json", "results/confusion_matrix.png", "results/threshold_tradeoff.png"):
        assert (trained_dir / f).exists()
