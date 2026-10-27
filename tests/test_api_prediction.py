
from pathlib import Path
import json
import urllib.request
import urllib.error

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/v3_2/xgboost.joblib"
TEST_PATH = ROOT / "data/v3_2/splits/test.csv"

API_URL = "http://127.0.0.1:8000"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["feature_columns"]
threshold = float(artifact["threshold"])

test = pd.read_csv(TEST_PATH)

# Choose a real feature record from our synthetic test set.
record = test.iloc[0]

feature_values = {
    name: float(record[name])
    for name in features
}

assert all(
    np.isfinite(value)
    for value in feature_values.values()
)

payload = json.dumps({
    "features": feature_values
}).encode("utf-8")

request = urllib.request.Request(
    url=f"{API_URL}/predict",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(
        request, timeout=10
    ) as response:
        result = json.load(response)

except urllib.error.URLError as error:
    raise SystemExit(
        f"API connection failed: {error}. "
        "Make sure Uvicorn is running on port 8000."
    )

# Independently calculate the expected prediction.
X = pd.DataFrame(
    [[feature_values[name] for name in features]],
    columns=features,
)

expected_score = float(
    model.predict_proba(X)[0, 1]
)

expected_alert = expected_score >= threshold

# Verify API and direct model predictions agree.
assert np.isclose(
    result["risk_score"],
    expected_score,
    rtol=1e-6,
    atol=1e-8,
)

assert result["alert"] == expected_alert
assert np.isclose(result["threshold"], threshold)
assert result["model_name"] == "xgboost"

print("\n=== SENTINELAI API PREDICTION TEST ===")

print("Event ID:", record["event_id"])
print("Scenario:", record["scenario_type"])
print("True synthetic label:", int(record["risk_label"]))
print("API risk score:", round(result["risk_score"], 6))
print("Model threshold:", round(result["threshold"], 6))
print("API alert:", result["alert"])
print("Direct model score:", round(expected_score, 6))

print("\nPASS: API prediction matches saved model")
print("PASS: API alert decision matches threshold")
print("PASS: Real HTTP request completed successfully")
