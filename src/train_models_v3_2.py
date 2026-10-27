
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


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/v3_2/splits"
OUTPUT_DIR = ROOT / "models/v3_2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALERTS_PER_1000 = 20
RANDOM_STATE = 42


# ==================================================
# 1. LOAD DATA
# ==================================================

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(
    DATA_DIR / "validation.csv"
)
test = pd.read_csv(DATA_DIR / "test.csv")

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
    "actor_events_last_1h",
    "actor_volume_last_1h_mb",
    "actor_unique_resources_last_1h",
    "actor_workflow_previous_events",
    "actor_workflow_volume_before_mb",
    "actor_resource_seen_before",
]

for name, df in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    assert df["event_id"].is_unique, name
    assert df["risk_label"].isin([0, 1]).all()
    assert df[FEATURE_COLUMNS].notna().all().all()
    assert np.isfinite(
        df[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all()

X_train = train[FEATURE_COLUMNS]
y_train = train["risk_label"]

X_val = validation[FEATURE_COLUMNS]
y_val = validation["risk_label"]

X_test = test[FEATURE_COLUMNS]
y_test = test["risk_label"]


# ==================================================
# 2. INITIALIZE MODELS
# ==================================================

positives = int(y_train.sum())
negatives = int(len(y_train) - positives)

assert positives > 0

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ]),

    "xgboost": XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negatives / positives,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
    ),
}


# ==================================================
# 3. EVALUATION HELPERS
# ==================================================

def choose_threshold(scores, alerts_per_1000):
    """
    Choose threshold using validation scores only.
    No labels or test scores are used for selection.
    """

    scores = np.asarray(scores, dtype=float)

    budget = max(
        1,
        int(len(scores) * alerts_per_1000 / 1000),
    )

    sorted_scores = np.sort(scores)[::-1]

    if budget >= len(scores):
        return float(np.nextafter(
            sorted_scores[-1], -np.inf
        ))

    upper = float(sorted_scores[budget - 1])
    lower = float(sorted_scores[budget])

    if upper > lower:
        return (upper + lower) / 2

    # A tied score cannot be split by a scalar threshold.
    # Include the entire tied group.
    return upper


def evaluate(y_true, scores, threshold):
    predictions = (
        np.asarray(scores) >= threshold
    ).astype(int)

    y_true = np.asarray(y_true)

    false_positives = int(
        ((predictions == 1) & (y_true == 0)).sum()
    )

    return {
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "threshold": float(threshold),
        "alerts": int(predictions.sum()),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "false_alerts_per_1000": float(
            false_positives / len(y_true) * 1000
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def scenario_report(df, scores, threshold):
    report = df[
        ["event_id", "incident_id",
         "scenario_type", "risk_label"]
    ].copy()

    report["prediction"] = (
        np.asarray(scores) >= threshold
    ).astype(int)

    grouped = report.groupby(
        "scenario_type"
    ).agg(
        events=("event_id", "count"),
        positives=("risk_label", "sum"),
        alerts=("prediction", "sum"),
    )

    risky = report[
        report["risk_label"] == 1
    ]

    detected = risky[
        risky["prediction"] == 1
    ]

    incident_total = (
        risky["incident_id"].dropna().nunique()
    )

    incidents_detected = (
        detected["incident_id"].dropna().nunique()
    )

    return {
        "scenario_counts": grouped.to_dict(
            orient="index"
        ),
        "risk_events_detected": int(
            len(detected)
        ),
        "total_risk_events": int(
            len(risky)
        ),
        "incidents_detected": int(
            incidents_detected
        ),
        "total_incidents": int(
            incident_total
        ),
    }


# ==================================================
# 4. TRAIN AND VALIDATE
# ==================================================

trained = {}
results = {}

print("\n=== V3.2 MODEL TRAINING ===")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    val_scores = model.predict_proba(
        X_val
    )[:, 1]

    threshold = choose_threshold(
        val_scores,
        ALERTS_PER_1000,
    )

    val_metrics = evaluate(
        y_val,
        val_scores,
        threshold,
    )

    trained[name] = {
        "model": model,
        "threshold": threshold,
    }

    results[name] = {
        "validation": val_metrics,
    }

    print(
        "Validation Average Precision:",
        round(val_metrics["average_precision"], 4),
    )

    print(
        "Validation recall:",
        round(val_metrics["recall"], 4),
    )

    print(
        "Validation alerts:",
        val_metrics["alerts"],
    )


# ==================================================
# 5. SELECT MODEL USING VALIDATION ONLY
# ==================================================

best_model_name = max(
    results,
    key=lambda name: (
        results[name]["validation"][
            "average_precision"
        ]
    ),
)

print("\nSelected model:", best_model_name)
print(
    "Selection criterion: "
    "Validation Average Precision"
)


# ==================================================
# 6. FINAL TEST EVALUATION
# ==================================================

selected = trained[best_model_name]

test_scores = selected[
    "model"
].predict_proba(X_test)[:, 1]

test_metrics = evaluate(
    y_test,
    test_scores,
    selected["threshold"],
)

scenario_metrics = scenario_report(
    test,
    test_scores,
    selected["threshold"],
)

results["selected_model"] = best_model_name
results["selected_test_metrics"] = test_metrics
results["selected_test_scenarios"] = (
    scenario_metrics
)

print("\n=== SELECTED MODEL TEST RESULTS ===")
print(json.dumps(test_metrics, indent=2))

print("\n=== TEST SCENARIO ANALYSIS ===")
print(json.dumps(scenario_metrics, indent=2))


# ==================================================
# 7. SAVE ARTIFACTS
# ==================================================

for name, item in trained.items():

    joblib.dump(
        {
            "model": item["model"],
            "feature_columns": FEATURE_COLUMNS,
            "threshold": item["threshold"],
            "alerts_per_1000": ALERTS_PER_1000,
            "dataset_version": "v3.2",
        },
        OUTPUT_DIR / f"{name}.joblib",
    )

with open(
    OUTPUT_DIR / "comparison.json",
    "w",
) as file:
    json.dump(results, file, indent=2)

print("\nPASS: V3.2 model training complete")
print("Saved to:", OUTPUT_DIR)
