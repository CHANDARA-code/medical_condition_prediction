from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

def _resolve_artifacts_dir() -> Path:
    env_dir = os.getenv("MODEL_ARTIFACTS_DIR", "").strip()
    candidates: list[Path] = []

    if env_dir:
        candidates.append(Path(env_dir))

    candidates.extend(
        [
            Path(__file__).resolve().parents[1] / "artifacts",
            Path(__file__).resolve().parent / "artifacts",
            Path.cwd() / "artifacts",
        ]
    )

    for candidate in candidates:
        model_file = candidate / "best_model.joblib"
        meta_file = candidate / "model_metadata.json"
        if model_file.exists() and meta_file.exists():
            return candidate

    searched = ", ".join(str(p) for p in candidates)
    raise RuntimeError(
        "Model artifacts not found. Expected `best_model.joblib` and `model_metadata.json`. "
        f"Searched: {searched}. Set MODEL_ARTIFACTS_DIR if needed."
    )


ARTIFACTS_DIR = _resolve_artifacts_dir()
model_path = ARTIFACTS_DIR / "best_model.joblib"
metadata_path = ARTIFACTS_DIR / "model_metadata.json"

model = joblib.load(model_path)
metadata = json.loads(metadata_path.read_text())
features: list[str] = metadata["features"]
classes: list[str] = metadata["classes"]


class PredictRequest(BaseModel):
    symptoms: List[str] = Field(default_factory=list, description="List of symptom codes")


class PredictionScore(BaseModel):
    condition: str
    probability: float


class PredictResponse(BaseModel):
    predicted_condition: str
    confidence: float
    top_predictions: List[PredictionScore]
    active_symptom_count: int


app = FastAPI(title="Medical Symptom Condition Predictor", version="1.0.0")


def _get_allowed_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if configured:
        return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://161.97.114.68:5173",
        "http://161.97.114.68:2343",
        "https://predictlogistic.leavchandara.site"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": metadata.get("best_model"),
        "features": len(features),
        "classes": len(classes),
    }


@app.get("/symptoms")
def list_symptoms() -> dict:
    return {
        "symptoms": [
            {"code": s, "label": s.replace("_", " ").title()} for s in sorted(features)
        ]
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not payload.symptoms:
        raise HTTPException(status_code=400, detail="Please provide at least one symptom")

    unknown = [s for s in payload.symptoms if s not in features]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown symptoms: {', '.join(unknown[:10])}",
        )

    row = {feature: 0 for feature in features}
    for symptom in payload.symptoms:
        row[symptom] = 1

    x = pd.DataFrame([row])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
    else:
        scores = model.decision_function(x)
        if scores.ndim > 1:
            scores = scores[0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

    top_idx = np.argsort(probs)[::-1][:3]
    top_predictions = [
        PredictionScore(condition=classes[i], probability=float(probs[i])) for i in top_idx
    ]

    return PredictResponse(
        predicted_condition=top_predictions[0].condition,
        confidence=top_predictions[0].probability,
        top_predictions=top_predictions,
        active_symptom_count=len(payload.symptoms),
    )
