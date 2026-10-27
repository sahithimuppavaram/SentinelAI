
import joblib
import numpy as np
import pandas as pd

# Load engineered features and labels
features = pd.read_csv(
    "data/v2/behavioral_features.csv",
    parse_dates=["event_time"]
)

labels = pd.read_csv("data/v2/labels.csv")

# Join labels for offline evaluation
data = features.merge(
    labels[["event_id", "risk_label", "scenario_type"]],
    on="event_id",
    validate="one_to_one"
)

# Reproduce the chronological test split
data = data.sort_values(
    ["event_time", "event_id"]
).reset_index(drop=True)

test = data.iloc[int(len(data) * 0.80):].copy()

# Load both model artifacts
lr_artifact = joblib.load(
    "models/v2/logistic_regression.joblib"
)

xgb_artifact = joblib.load(
    "models/v2/xgboost.joblib"
)

for name, artifact in [
    ("lr", lr_artifact),
    ("xgb", xgb_artifact)
]:
    model = artifact["model"]
    columns = artifact["feature_columns"]
    threshold = artifact["threshold"]

    probabilities = model.predict_proba(
        test[columns]
    )[:, 1]

    test[f"{name}_score"] = probabilities

    test[f"{name}_prediction"] = (
        probabilities >= threshold
    ).astype(int)

# Focus on XGBoost's mistakes
false_positives = test[
    (test["risk_label"] == 0)
    & (test["xgb_prediction"] == 1)
]

false_negatives = test[
    (test["risk_label"] == 1)
    & (test["xgb_prediction"] == 0)
]

print("\n=== XGBOOST ERROR COUNTS ===")
print("False positives:", len(false_positives))
print("False negatives:", len(false_negatives))

print("\n=== FALSE POSITIVES BY SCENARIO ===")
print(false_positives["scenario_type"].value_counts())

columns_to_show = [
    "event_id",
    "actor_id",
    "scenario_type",
    "size_mb",
    "destination_type",
    "actor_size_ratio",
    "actor_events_last_24h",
    "xgb_score",
]

print("\n=== TOP 10 FALSE POSITIVES ===")

print(
    false_positives
    .sort_values("xgb_score", ascending=False)
    [columns_to_show]
    .head(10)
    .to_string(index=False)
)

print("\n=== TOP 10 MISSED RISK EVENTS ===")

print(
    false_negatives
    .sort_values("xgb_score", ascending=True)
    [columns_to_show]
    .head(10)
    .to_string(index=False)
)

# Compare which risky events each model detected
risky = test[test["risk_label"] == 1]

print("\n=== DETECTION OVERLAP ===")

print(pd.crosstab(
    risky["lr_prediction"],
    risky["xgb_prediction"],
    rownames=["Logistic Regression"],
    colnames=["XGBoost"]
))

print("\n=== ERROR ANALYSIS COMPLETE ===")
