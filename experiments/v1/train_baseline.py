
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. Load features and labels
features = pd.read_csv(
    "data/features.csv",
    parse_dates=["event_time"]
)
labels = pd.read_csv("data/labels.csv")

# 2. Join features with labels for training
data = features.merge(
    labels[["event_id", "risk_label"]],
    on="event_id",
    validate="one_to_one"
)

# 3. Sort chronologically
data = data.sort_values(
    ["event_time", "event_id"]
).reset_index(drop=True)

# 4. Choose model input columns
feature_columns = [
    "event_hour_utc",
    "event_day_of_week",
    "log_size_mb",
    "is_high_sensitivity",
    "is_unapproved_destination",
    "is_upload",
    "is_ai_agent",
]

# 5. Chronological 60/20/20 split
n = len(data)
train_end = int(n * 0.60)
validation_end = int(n * 0.80)

train = data.iloc[:train_end]
validation = data.iloc[train_end:validation_end]
test = data.iloc[validation_end:]

X_train = train[feature_columns]
y_train = train["risk_label"]

X_val = validation[feature_columns]
y_val = validation["risk_label"]

X_test = test[feature_columns]
y_test = test["risk_label"]

# 6. Create model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# 7. Train only on training data
model.fit(X_train, y_train)

# 8. Use validation data to select threshold
val_probs = model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.05, 0.95, 91)

results = []

for threshold in thresholds:
    predictions = (
        val_probs >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    results.append((f1, threshold))

best_f1, best_threshold = max(results)

# 9. Evaluate once on held-out test data
test_probs = model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probs >= best_threshold
).astype(int)

metrics = {
    "train_rows": len(train),
    "validation_rows": len(validation),
    "test_rows": len(test),
    "selected_threshold": float(best_threshold),
    "precision": float(precision_score(
        y_test, test_predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, test_predictions, zero_division=0
    )),
    "pr_auc": float(average_precision_score(
        y_test, test_probs
    )),
    "roc_auc": float(roc_auc_score(
        y_test, test_probs
    )),
}

print("\n=== BASELINE RESULTS ===")
print(json.dumps(metrics, indent=2))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, test_predictions))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    y_test, test_predictions, zero_division=0
))

# 10. Save artifacts
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "threshold": float(best_threshold)
    },
    output_dir / "baseline.joblib"
)

with open(output_dir / "baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nModel saved to models/baseline.joblib")
print("Metrics saved to models/baseline_metrics.json")
