
from pathlib import Path
import json
import urllib.request
import uuid

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

test = pd.read_csv(
    ROOT / "data/v3_2/splits/test.csv"
)

artifact = joblib.load(
    ROOT / "models/v3_2/xgboost.joblib"
)

model = artifact["model"]
columns = artifact["feature_columns"]
threshold = float(artifact["threshold"])

# Find one event that raises an alert and
# one event that does not.
scores = model.predict_proba(
    test[columns]
)[:, 1]

alert_indices = [
    i for i, score in enumerate(scores)
    if score >= threshold
]

normal_indices = [
    i for i, score in enumerate(scores)
    if score < threshold
]

assert alert_indices
assert normal_indices


def submit_event(index):
    row = test.iloc[index]

    # Use a fresh ID for each API test.
    # Avoid collisions with imported event IDs.
    event_id = f"API-{uuid.uuid4().hex[:20]}"

    features = {
        name: float(row[name])
        for name in columns
    }

    payload = {
        "event_id": event_id,
        "actor_id": str(row["actor_id"]),
        "actor_type": str(row["actor_type"]),
        "event_time": str(row["event_time"]),
        "action": str(row["action"]),
        "resource_id": str(row["resource_id"]),
        "destination_type": str(
            row["destination_type"]
        ),
        "size_mb": float(row["size_mb"]),
        "features": features,
    }

    request = urllib.request.Request(
        "http://127.0.0.1:8000/events/score",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request, timeout=10
    ) as response:
        result = json.load(response)

    assert result["saved"] is True
    assert result["event_id"] == event_id
    assert result["prediction_id"] > 0

    expected_alert = (
        scores[index] >= threshold
    )

    assert result["alert"] == expected_alert

    if expected_alert:
        assert result["alert_id"] is not None
    else:
        assert result["alert_id"] is None

    print("\nEvent ID:", event_id)
    print("Score:", round(result["risk_score"], 6))
    print("Alert:", result["alert"])
    print("Prediction ID:", result["prediction_id"])
    print("Alert ID:", result["alert_id"])


print("\n=== POSTGRESQL INTEGRATION TEST ===")

print("\n1. Testing non-alert event")
submit_event(normal_indices[0])

print("\n2. Testing alert event")
submit_event(alert_indices[0])

print("\nPASS: Both events saved through FastAPI")
print("PASS: Predictions returned database IDs")
print("PASS: Alert created only when required")
