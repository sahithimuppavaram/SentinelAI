
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.database import get_connection


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/v3_2/xgboost.joblib"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_columns = artifact["feature_columns"]
threshold = float(artifact["threshold"])

app = FastAPI(
    title="SentinelAI Prediction API",
    version="0.2.0",
    description="Experimental insider-risk scoring service",
)


class PredictionRequest(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    risk_score: float
    threshold: float
    alert: bool
    model_name: str


class EventScoreRequest(BaseModel):
    event_id: str = Field(min_length=1, max_length=50)
    actor_id: str = Field(min_length=1, max_length=50)
    actor_type: str = Field(
        pattern="^(human|ai_agent)$"
    )
    event_time: datetime
    action: str = Field(min_length=1, max_length=50)
    resource_id: str | None = Field(
        default=None, max_length=100
    )
    destination_type: str | None = Field(
        default=None, max_length=50
    )
    size_mb: float = Field(ge=0, allow_inf_nan=False)
    features: dict[str, float]


class EventScoreResponse(PredictionResponse):
    event_id: str
    prediction_id: int
    alert_id: int | None = None
    saved: bool


def calculate_prediction(features: dict[str, float]):
    missing = [
        name for name in feature_columns
        if name not in features
    ]

    unexpected = [
        name for name in features
        if name not in feature_columns
    ]

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": unexpected,
            },
        )

    if not all(
        np.isfinite(features[name])
        for name in feature_columns
    ):
        raise HTTPException(
            status_code=422,
            detail="All feature values must be finite",
        )

    X = pd.DataFrame(
        [[features[name] for name in feature_columns]],
        columns=feature_columns,
    )

    score = float(model.predict_proba(X)[0, 1])

    return {
        "risk_score": score,
        "threshold": threshold,
        "alert": score >= threshold,
        "model_name": "xgboost",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.get("/model/info")
def model_info():
    return {
        "model_name": "xgboost",
        "dataset_version": "v3.2",
        "feature_count": len(feature_columns),
        "required_features": feature_columns,
        "threshold": threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return calculate_prediction(request.features)


@app.post(
    "/events/score",
    response_model=EventScoreResponse,
)
def score_and_save_event(request: EventScoreRequest):

    prediction = calculate_prediction(request.features)

    try:
        # The connection manages a transaction:
        # commit on success, rollback on exception.
        with get_connection() as conn:
            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    INSERT INTO events (
                        event_id,
                        actor_id,
                        actor_type,
                        event_time,
                        action,
                        resource_id,
                        destination_type,
                        size_mb
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    """,
                    (
                        request.event_id,
                        request.actor_id,
                        request.actor_type,
                        request.event_time,
                        request.action,
                        request.resource_id,
                        request.destination_type,
                        request.size_mb,
                    ),
                )

                cursor.execute(
                    """
                    INSERT INTO predictions (
                        event_id,
                        model_name,
                        model_version,
                        risk_score,
                        threshold,
                        is_alert
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING prediction_id
                    """,
                    (
                        request.event_id,
                        "xgboost",
                        "v3.2",
                        prediction["risk_score"],
                        threshold,
                        prediction["alert"],
                    ),
                )

                prediction_id = cursor.fetchone()[0]
                alert_id = None

                if prediction["alert"]:
                    cursor.execute(
                        """
                        INSERT INTO alerts (
                            prediction_id,
                            status
                        )
                        VALUES (%s, 'open')
                        RETURNING alert_id
                        """,
                        (prediction_id,),
                    )

                    alert_id = cursor.fetchone()[0]

    except psycopg.errors.UniqueViolation:
        raise HTTPException(
            status_code=409,
            detail="Event ID already exists",
        )

    except (psycopg.OperationalError, RuntimeError):
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )

    return EventScoreResponse(
        event_id=request.event_id,
        prediction_id=prediction_id,
        alert_id=alert_id,
        saved=True,
        **prediction,
    )
