
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.behavioral_features import build_behavioral_features


DATA_DIR = Path("data/v3")


def test_dataset_integrity():
    events = pd.read_csv(DATA_DIR / "events.csv")
    features = pd.read_csv(
        DATA_DIR / "contextual_features.csv"
    )

    assert len(events) == len(features)
    assert events["event_id"].is_unique
    assert features["event_id"].is_unique

    assert set(events["event_id"]) == set(
        features["event_id"]
    )

    print("PASS: Event and feature records match")


def test_no_invalid_numerical_features():
    features = pd.read_csv(
        DATA_DIR / "contextual_features.csv"
    )

    columns = [
        "log_size_mb",
        "actor_previous_events",
        "actor_previous_avg_size",
        "actor_size_ratio",
        "actor_events_last_24h",
        "is_cold_start",
        "is_unauthorized",
        "is_ai_agent",
    ]

    values = features[columns].to_numpy(dtype=float)

    assert np.isfinite(values).all()

    print("PASS: Numerical features are finite")


def test_historical_behavior():
    events = pd.DataFrame([
        {
            "event_id": "E1",
            "event_time": "2026-09-01T10:00:00Z",
            "actor_id": "A001",
            "size_mb": 10.0,
            "destination_type": "internal",
            "sensitivity": "internal",
        },
        {
            "event_id": "E2",
            "event_time": "2026-09-01T10:00:00Z",
            "actor_id": "A001",
            "size_mb": 20.0,
            "destination_type": "internal",
            "sensitivity": "internal",
        },
        {
            "event_id": "E3",
            "event_time": "2026-09-01T11:00:00Z",
            "actor_id": "A001",
            "size_mb": 30.0,
            "destination_type": "internal",
            "sensitivity": "internal",
        },
    ])

    result = build_behavioral_features(events)
    result = result.set_index("event_id")

    assert result.loc["E1", "actor_previous_events"] == 0
    assert result.loc["E2", "actor_previous_events"] == 0

    assert result.loc["E3", "actor_previous_events"] == 2

    assert np.isclose(
        result.loc["E3", "actor_previous_avg_size"],
        15.0
    )

    assert result.loc["E3", "actor_events_last_24h"] == 2

    print("PASS: Historical features exclude simultaneous events")


if __name__ == "__main__":
    test_dataset_integrity()
    test_no_invalid_numerical_features()
    test_historical_behavior()

    print("\nPASS: All V3 contextual feature tests")
