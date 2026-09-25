import json
from pathlib import Path
import pytest
from app.schemas import FEATURES

EX = Path(__file__).resolve().parent.parent / "examples"
malignant = json.loads((EX / "malignant_sample.json").read_text())
benign = json.loads((EX / "benign_sample.json").read_text())


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model-info").json()
    assert r["n_features"] == 30 and r["feature_names"] == FEATURES
    assert 0 < r["threshold"] <= 0.5 and "disclaimer" in r


def test_predict_malignant_and_benign_samples(client):
    a, b = client.post("/predict", json=malignant).json(), client.post("/predict", json=benign).json()
    assert a["label"] == "malignant" and b["label"] == "benign"
    assert a["probability_malignant"] > b["probability_malignant"]
    assert a["probability_malignant"] + a["probability_benign"] == pytest.approx(1, abs=1e-3)


def test_predict_batch_keeps_order(client):
    r = client.post("/predict-batch", json={"samples": [malignant, benign, malignant]}).json()
    assert [p["label"] for p in r["predictions"]] == ["malignant", "benign", "malignant"]


@pytest.mark.parametrize("bad", [
    {k: v for k, v in benign.items() if k != "mean_radius"},          # missing field
    {**benign, "mean_radius": -1},                                      # negative value
    {**benign, "mean_radius": "big"},                                   # wrong type
    {**benign, "not_a_feature": 1.0},                                   # unknown extra field
])
def test_bad_input_is_rejected_with_422(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_limits(client):
    assert client.post("/predict-batch", json={"samples": []}).status_code == 422
    assert client.post("/predict-batch", json={"samples": [benign] * 101}).status_code == 422


def test_missing_model_fails_loudly_at_startup(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="python train.py"):
        with TestClient(app):
            pass
