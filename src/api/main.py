"""
FastAPI prediction API.
Exposes a /predict endpoint that runs the full feature pipeline + ML inference.
Integrates with the Observer event bus for monitoring and alerting.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ingestion.source_factory import SourceFactory
from src.models.model_registry import ModelRegistry, BaseModel as MLModel
from src.monitoring.event_bus import Event, EventBus, EventType, HighRiskAlertHandler, LoggingHandler, ModelDriftHandler
from src.processing.features import FeaturePipeline, GeoSample

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# App State
# --------------------------------------------------------------------------- #

class AppState:
    model: MLModel | None = None
    feature_pipeline: FeaturePipeline | None = None
    event_bus: EventBus | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ML model, feature pipeline, and event bus on startup."""
    logger.info("Starting BellwetherML API...")

    # Event bus + observers
    state.event_bus = EventBus()
    state.event_bus.subscribe(EventType.PREDICTION_MADE, LoggingHandler())
    state.event_bus.subscribe(EventType.PREDICTION_MADE, HighRiskAlertHandler(threshold=0.75))
    state.event_bus.subscribe(EventType.PREDICTION_MADE, ModelDriftHandler(window=100))

    # Feature pipeline
    state.feature_pipeline = FeaturePipeline.default()

    # Model
    model_path = Path(os.getenv("MODEL_PATH", "models/xgboost.pkl"))
    model_name = os.getenv("MODEL_NAME", "random_forest")

    if model_path.exists():
        state.model = ModelRegistry.create(model_name)
        logger.info("Loaded model from %s", model_path)
    else:
        logger.warning("No saved model found at %s — training on mock data", model_path)
        import pandas as pd
        source = SourceFactory.create("mock", n_samples=2000)
        samples = list(source.stream())
        X = state.feature_pipeline.transform_batch(samples)
        import random
        rng = random.Random(42)
        y = pd.Series([rng.choice([0, 0, 0, 1]) for _ in samples])
        state.model = ModelRegistry.create(model_name)
        state.model.train(X, y)

    yield
    logger.info("Shutting down BellwetherML API.")


app = FastAPI(
    title="BellwetherML — Wildfire Risk API",
    description="Geospatial ML pipeline for real-time wildfire risk prediction.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    ndvi: float = Field(..., ge=-1, le=1, description="Normalized Difference Vegetation Index")
    land_surface_temp: float = Field(..., description="Land surface temperature in Kelvin")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity %")
    elevation: float = Field(0.0, description="Elevation in meters")
    slope: float = Field(0.0, ge=0, le=90, description="Terrain slope in degrees")
    days_since_rain: int = Field(0, ge=0)
    historical_fire: int = Field(0, ge=0, le=1)


class PredictResponse(BaseModel):
    risk_score: float
    risk_label: str
    features_used: dict[str, float]
    model: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": state.model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if state.model is None or state.feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    sample = GeoSample(**req.model_dump())
    features = state.feature_pipeline.transform(sample)

    import pandas as pd
    X = pd.DataFrame([features])
    score = float(state.model.predict_proba(X)[0])

    label = "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.40 else "LOW"

    state.event_bus.publish(Event(
        type=EventType.PREDICTION_MADE,
        payload={"risk_score": score, "location": f"{req.latitude},{req.longitude}"},
        source="predict_endpoint",
    ))

    return PredictResponse(
        risk_score=round(score, 4),
        risk_label=label,
        features_used=features,
        model=state.model.__class__.__name__,
    )


@app.get("/models")
async def list_models():
    return {"available": ModelRegistry.available()}
