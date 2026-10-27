
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ===========================================
# CONFIGURATION
# ===========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/v3_2/splits"
OUTPUT_DIR = ROOT / "models/v3_2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
ALERTS_PER_1000 = 20

# Focus on behavioral signals rather than
# explicit authorization-policy indicators.
FEATURE_COLUMNS = [
    "log_size_mb",
    "event_hour_utc",
    "actor_previous_events",
    "actor_previous_avg_size",
    "actor_size_ratio",
    "actor_events_last_24h",
    "actor_events_last_1h",
    "actor_volume_last_1h_mb",
    "actor_unique_resources_last_1h",
    "actor_workflow_previous_events",
    "actor_workflow_volume_before_mb",
    "actor_resource_seen_before",
]


# ===========================================
# 1. LOAD DATA
# ===========================================

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(
    DATA_DIR / "validation.csv"
)
test = pd.read_csv(DATA_DIR / "test.csv")

for name, df in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    assert df["event_id"].is_unique, name

    values = df[FEATURE_COLUMNS].to_numpy(
        dtype=float
    )

    assert np.isfinite(values).all(), name

X_train = train[FEATURE_COLUMNS]
X_val = validation[FEATURE_COLUMNS]
X_test = test[FEATURE_COLUMNS]

y_val = validation["risk_label"].to_numpy()
y_test = test["risk_label"].to_numpy()


# ===========================================
# 2. BUILD ISOLATION FOREST
# ===========================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("detector", IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

print("\n=== V3.2 ISOLATION FOREST ===")

print("Training events:", len(train))
print("Validation events:", len(validation))
print("Test events:", len(test))

# Fit without using risk labels.
model.fit(X_train)


# ===========================================
# 3. ANOMALY SCORING
# ===========================================

def anomaly_scores(X):
    """
    Higher score = more anomalous.

    scikit-learn decision_function produces
    lower scores for more abnormal events,
    so reverse its sign.
    """
    return -model.decision_function(X)


val_scores = anomaly_scores(X_val)


# ===========================================
# 4. SET THRESHOLD USING VALIDATION ONLY
# ===========================================

def choose_threshold(scores, alerts_per_1000):
    scores = np.asarray(scores, dtype=float)

    budget = max(
        1,
        int(
            len(scores)
            * alerts_per_1000
            / 1000
        ),
    )

    sorted_scores = np.sort(scores)[::-1]

    if budget >= len(scores):
        return float(
            np.nextafter(
                sorted_scores[-1],
                -np.inf,
            )
        )

    upper = float(sorted_scores[budget - 1])
    lower = float(sorted_scores[budget])

    if upper > lower:
        return (upper + lower) / 2.0

    # A threshold cannot split tied scores.
    return upper


threshold = choose_threshold(
    val_scores,
    ALERTS_PER_1000,
)


# ===========================================
# 5. EVALUATION
# ===========================================

def evaluate(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    predictions = (
        scores >= threshold
    ).astype(int)

    false_positives = int(
        (
            (predictions == 1)
            & (y_true == 0)
        ).sum()
    )

    return {
        "average_precision": float(
            average_precision_score(
                y_true, scores
            )
        ),
        "alerts": int(predictions.sum()),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "false_alerts_per_1000": float(
            false_positives
            / len(y_true)
            * 1000
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }


validation_metrics = evaluate(
    y_val,
    val_scores,
    threshold,
)

print("\nValidation results:")
print(json.dumps(validation_metrics, indent=2))

# Evaluate the frozen model and threshold.
test_scores = anomaly_scores(X_test)

test_metrics = evaluate(
    y_test,
    test_scores,
    threshold,
)

print("\nTest results:")
print(json.dumps(test_metrics, indent=2))


# ===========================================
# 6. ANALYZE ALERT SCENARIOS
# ===========================================

test_results = test[
    [
        "event_id",
        "scenario_type",
        "incident_id",
        "risk_label",
    ]
].copy()

test_results["anomaly_score"] = test_scores

test_results["alert"] = (
    test_scores >= threshold
).astype(int)

print("\nAlerts by scenario:")

print(
    test_results.groupby(
        "scenario_type"
    )["alert"].agg(
        ["count", "sum"]
    ).to_string()
)

risky = test_results[
    test_results["risk_label"] == 1
]

detected = risky[risky["alert"] == 1]

print(
    "\nIncidents detected:",
    detected["incident_id"].dropna().nunique(),
    "/",
    risky["incident_id"].dropna().nunique(),
)


# ===========================================
# 7. SAVE MODEL AND METRICS
# ===========================================

artifact_path = (
    OUTPUT_DIR / "isolation_forest.joblib"
)

joblib.dump(
    {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "dataset_version": "v3.2",
        "score_direction": "higher_is_more_anomalous",
    },
    artifact_path,
)

results = {
    "algorithm": "IsolationForest",
    "training": "Unsupervised on full training data",
    "threshold": float(threshold),
    "validation": validation_metrics,
    "test": test_metrics,
}

with open(
    OUTPUT_DIR / "anomaly_comparison.json",
    "w",
) as file:
    json.dump(
        results,
        file,
        indent=2,
    )

print("\nPASS: Isolation Forest experiment complete")
print("Saved to:", OUTPUT_DIR)
