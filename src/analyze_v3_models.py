
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DATA_DIR = Path("data/v3")
MODEL_DIR = Path("models/v3")

# Load test data and scenario information
test = pd.read_csv(DATA_DIR / "splits/test.csv")

labels = pd.read_csv(
    DATA_DIR / "event_labels.csv"
)

incidents = pd.read_csv(
    DATA_DIR / "incidents.csv"
)

test = test.merge(
    labels[["event_id", "scenario_type"]],
    on="event_id",
    validate="one_to_one"
)

# Ensure test events are unique
assert test["event_id"].is_unique

print("\n=== V3 INCIDENT DETECTION ANALYSIS ===")

for name in ["logistic_regression", "xgboost"]:

    artifact = joblib.load(
        MODEL_DIR / f"{name}.joblib"
    )

    model = artifact["model"]
    columns = artifact["feature_columns"]

    probabilities = model.predict_proba(
        test[columns]
    )[:, 1]

    # Reproduce the offline top-50 alert selection
    budget = max(
        1,
        int(len(test) * 20 / 1000)
    )

    ranked = np.argsort(
        -probabilities,
        kind="stable"
    )

    predictions = np.zeros(
        len(test), dtype=int
    )

    predictions[ranked[:budget]] = 1

    result = test.copy()
    result["risk_score"] = probabilities
    result["prediction"] = predictions

    print(f"\n=== {name.upper()} ===")

    # Detection by risk scenario
    risky = result[result["risk_label"] == 1]

    print("\nDetection by scenario:")

    summary = risky.groupby(
        "scenario_type"
    ).agg(
        total_events=("event_id", "count"),
        detected_events=("prediction", "sum")
    )

    summary["recall"] = (
        summary["detected_events"]
        / summary["total_events"]
    )

    print(summary.to_string())

    # Incident-level detection
    detected_incidents = set(
        risky.loc[
            risky["prediction"] == 1,
            "incident_id"
        ].dropna()
    )

    all_test_incidents = set(
        risky["incident_id"].dropna()
    )

    print("\nIncident-level results:")
    print("Total incidents:", len(all_test_incidents))
    print("Detected incidents:", len(detected_incidents))
    print(
        "Missed incidents:",
        len(all_test_incidents - detected_incidents)
    )

    # False positives by scenario
    false_positives = result[
        (result["risk_label"] == 0)
        & (result["prediction"] == 1)
    ]

    print("\nFalse positives by scenario:")
    print(
        false_positives[
            "scenario_type"
        ].value_counts().to_string()
    )

    # Check reliance on authorization
    print("\nDetection by authorization status:")

    print(
        pd.crosstab(
            risky["authorization_status"],
            risky["prediction"],
            rownames=["Authorization"],
            colnames=["Predicted risky"],
        )
    )

print("\n=== ANALYSIS COMPLETE ===")
