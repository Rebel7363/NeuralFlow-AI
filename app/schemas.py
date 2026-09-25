"""Request/response shapes. FastAPI uses these to validate input and to build the /docs page."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, create_model

# The 30 measurements, in the exact order the model was trained on.
FEATURES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension",
]

# A real (benign) row from the dataset, shown as the example in /docs.
EXAMPLE = {"mean_radius": 13.54, "mean_texture": 14.36, "mean_perimeter": 87.46, "mean_area": 566.3, "mean_smoothness": 0.09779, "mean_compactness": 0.08129, "mean_concavity": 0.06664, "mean_concave_points": 0.04781, "mean_symmetry": 0.1885, "mean_fractal_dimension": 0.05766, "radius_error": 0.2699, "texture_error": 0.7886, "perimeter_error": 2.058, "area_error": 23.56, "smoothness_error": 0.008462, "compactness_error": 0.0146, "concavity_error": 0.02387, "concave_points_error": 0.01315, "symmetry_error": 0.0198, "fractal_dimension_error": 0.0023, "worst_radius": 15.11, "worst_texture": 19.26, "worst_perimeter": 99.7, "worst_area": 711.2, "worst_smoothness": 0.144, "worst_compactness": 0.1773, "worst_concavity": 0.239, "worst_concave_points": 0.1288, "worst_symmetry": 0.2977, "worst_fractal_dimension": 0.07259}

DISCLAIMER = "Educational project. Not a medical device. Never use it for a real diagnosis."

# One field per measurement. Every value must be a finite number >= 0 (all 30 features are non-negative),
# and unknown extra fields are rejected so typos do not slip through silently.
CancerFeatures = create_model(
    "CancerFeatures",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE}),
    **{name: (float, Field(..., ge=0, allow_inf_nan=False)) for name in FEATURES},
)


class BatchRequest(BaseModel):
    samples: list[CancerFeatures] = Field(..., min_length=1, max_length=100)


class Prediction(BaseModel):
    label: Literal["malignant", "benign"]
    probability_malignant: float
    probability_benign: float
    threshold: float = Field(..., description="Flag 'malignant' when probability_malignant >= threshold")
    model: str
    disclaimer: str = DISCLAIMER


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class ModelInfo(BaseModel):
    model_name: str
    threshold: float
    n_features: int
    feature_names: list[str]
    sklearn_version: str
    metrics: dict
    disclaimer: str = DISCLAIMER
