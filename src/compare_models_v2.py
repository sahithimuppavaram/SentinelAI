
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# Load features and labels
features = pd.read_csv(
    "data/v2/behavioral_features.csv",
    parse_dates=["event_time"]
)

labels = pd.read_csv("data/v2/labels.csv")

data = features.merge(
    labels[["event_id", "risk_label"]],
    on="event_id",
    validate="one_to_one"
)

data = data.sort_values(
    ["event_time", "event_id"]
).reset_index(drop=True)


# Features known at inference time
feature_columns = [
    "event_hour_utc",
    "log_size_mb",
    "is_high_sensitivity",
    "is_unapproved_destination",
    "actor_previous_events",
    "actor_previous_avg_size",
    "actor_size_ratio",
    "actor_events_last_24h",
    "is_cold_start",
]


# Chronological split
n = len(data)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = data.iloc[:train_end]
val = data.iloc[train_end:val_end]
test = data.iloc[val_end:]


X_train = train[feature_columns]
y_train = train["risk_label"]

X_val = val[feature_columns]
y_val = val["risk_label"]

X_test = test[feature_columns]
y_test = test["risk_label"]


# Models
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),

    "xgboost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4
    )
}


def select_threshold(y_true, probabilities):
    thresholds = np.linspace(0.05, 0.95, 91)

    scores = [
        (
            f1_score(
                y_true,
                probabilities >= threshold,
                zero_division=0
            ),
            float(threshold)
        )
        for threshold in thresholds
    ]

    return max(scores)[1]


Path("models/v2").mkdir(parents=True, exist_ok=True)

results = {}

for name, model in models.items():
    print(f"\n=== TRAINING {name.upper()} ===")

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]

    threshold = select_threshold(y_val, val_probs)

    test_probs = model.predict_proba(X_test)[:, 1]
    predictions = (test_probs >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_test, predictions, zero_division=0
        )),
        "pr_auc": float(average_precision_score(
            y_test, test_probs
        )),
        "roc_auc": float(roc_auc_score(
            y_test, test_probs
        )),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }

    results[name] = metrics

    print(json.dumps(metrics, indent=2))

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "threshold": threshold
        },
        f"models/v2/{name}.joblib"
    )


with open("models/v2/comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== MODEL COMPARISON COMPLETE ===")
