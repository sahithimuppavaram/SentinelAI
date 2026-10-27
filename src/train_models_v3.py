
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
    confusion_matrix,
)

from xgboost import XGBClassifier


# --------------------------------------
# 1. Load chronological data splits
# --------------------------------------

DATA_DIR = Path("data/v3/splits")
OUTPUT_DIR = Path("models/v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")


# --------------------------------------
# 2. Select inference-time features
# --------------------------------------

FEATURE_COLUMNS = [
    "log_size_mb",
    "event_hour_utc",
    "event_day_of_week",
    "is_high_sensitivity",
    "is_unapproved_destination",
    "is_unauthorized",
    "is_ai_agent",
    "actor_previous_events",
    "actor_previous_avg_size",
    "actor_size_ratio",
    "actor_events_last_24h",
    "is_cold_start",
]

X_train = train[FEATURE_COLUMNS]
y_train = train["risk_label"]

X_val = validation[FEATURE_COLUMNS]
y_val = validation["risk_label"]

X_test = test[FEATURE_COLUMNS]
y_test = test["risk_label"]


# --------------------------------------
# 3. Define ML models
# --------------------------------------

positive_count = int(y_train.sum())
negative_count = int(len(y_train) - positive_count)

scale_pos_weight = (
    negative_count / max(positive_count, 1)
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ]),

    "xgboost": XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    ),
}


# --------------------------------------
# 4. Evaluate under an alert budget
# --------------------------------------

def evaluate_at_budget(
    y_true,
    probabilities,
    alerts_per_1000=20,
):
    """
    Rank events by model score and select
    the highest-scoring events under a
    fixed analyst alert budget.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    budget = max(
        1,
        int(len(y_true) * alerts_per_1000 / 1000)
    )

    ranked_indices = np.argsort(
        -probabilities,
        kind="stable"
    )

    predictions = np.zeros(
        len(y_true),
        dtype=int
    )

    predictions[ranked_indices[:budget]] = 1

    return {
        "alerts": int(predictions.sum()),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "false_alerts_per_1000": float(
            ((predictions == 1) & (y_true == 0)).sum()
            / len(y_true) * 1000
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


# --------------------------------------
# 5. Train and evaluate each model
# --------------------------------------

results = {}

for name, model in models.items():

    print(f"\n=== {name.upper()} ===")

    model.fit(X_train, y_train)

    validation_scores = model.predict_proba(
        X_val
    )[:, 1]

    test_scores = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "validation_average_precision": float(
            average_precision_score(
                y_val, validation_scores
            )
        ),
        "test_average_precision": float(
            average_precision_score(
                y_test, test_scores
            )
        ),
        "test_at_alert_budget": evaluate_at_budget(
            y_test,
            test_scores,
            alerts_per_1000=20,
        ),
    }

    results[name] = metrics

    print(json.dumps(metrics, indent=2))

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "alert_budget_per_1000": 20,
        },
        OUTPUT_DIR / f"{name}.joblib",
    )


# --------------------------------------
# 6. Save experiment results
# --------------------------------------

with open(
    OUTPUT_DIR / "comparison.json",
    "w",
) as f:
    json.dump(results, f, indent=2)

print("\n=== V3 MODEL TRAINING COMPLETE ===")
print("Saved models and metrics to:", OUTPUT_DIR)
