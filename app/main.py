"""FastAPI service that serves the trained model.   Run:  uvicorn app.main:app --reload
Docs page (auto-generated): http://127.0.0.1:8000/docs"""
import json, logging, os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import sklearn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from .schemas import (FEATURES, BatchRequest, BatchResponse, CancerFeatures, Health, ModelInfo, Prediction)

log = logging.getLogger("neuralflow")


def load_artifacts(model_dir):
    """Load the model + metadata once, and fail loudly (at startup, not at the first request) if anything is off."""
    model_dir = Path(model_dir)
    try:
        model = joblib.load(model_dir / "model.joblib")   # only load files YOU trained (pickle is not safe for unknown files)
        meta = json.loads((model_dir / "meta.json").read_text())
    except FileNotFoundError as e:
        raise RuntimeError(f"Model files not found in '{model_dir}'. Run `python train.py` first.") from e
    if [n.replace(" ", "_") for n in meta["feature_names"]] != FEATURES:
        raise RuntimeError("Feature order in meta.json does not match the API schema.")
    if list(model.classes_) != [0, 1]:
        raise RuntimeError("Unexpected class order in the model.")
    if meta.get("sklearn_version") != sklearn.__version__:
        log.warning("Model trained with scikit-learn %s but %s is installed; retrain if results look odd.",
                    meta.get("sklearn_version"), sklearn.__version__)
    return model, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.meta = load_artifacts(os.getenv("MODEL_DIR", "models"))
    yield


app = FastAPI(
    title="NeuralFlow-AI",
    description="Predicts whether a breast-tumour measurement set looks **malignant** or **benign**. "
                "Educational project, not a medical device.",
    version="1.0.0",
    lifespan=lifespan,
)


def predict_rows(request: Request, rows) -> list[Prediction]:
    model, meta = request.app.state.model, request.app.state.meta
    X = np.array([[getattr(r, f) for f in FEATURES] for r in rows], dtype=float)  # fixed column order
    p_mal = model.predict_proba(X)[:, 0]                                           # class 0 = malignant
    thr = meta["threshold"]
    return [Prediction(label="malignant" if p >= thr else "benign",
                       probability_malignant=round(float(p), 4), probability_benign=round(float(1 - p), 4),
                       threshold=thr, model=meta["model_name"]) for p in p_mal]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=Health, tags=["ops"])
def health(request: Request):
    """Used by Docker / load balancers to check the service is alive."""
    return Health(status="ok", model_loaded=hasattr(request.app.state, "model"))


@app.get("/model-info", response_model=ModelInfo, tags=["ops"])
def model_info(request: Request):
    m = request.app.state.meta
    return ModelInfo(model_name=m["model_name"], threshold=m["threshold"], n_features=len(FEATURES),
                     feature_names=FEATURES, sklearn_version=m["sklearn_version"], metrics=m["metrics"])


@app.post("/predict", response_model=Prediction, tags=["predict"])
def predict(sample: CancerFeatures, request: Request):
    """Classify ONE sample (30 named measurements)."""
    return predict_rows(request, [sample])[0]


@app.post("/predict-batch", response_model=BatchResponse, tags=["predict"])
def predict_batch(batch: BatchRequest, request: Request):
    """Classify up to 100 samples in one call."""
    return BatchResponse(predictions=predict_rows(request, batch.samples))
