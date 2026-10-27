
import joblib
import pandas as pd

# Load trained model and settings
artifact = joblib.load("models/baseline.joblib")

model = artifact["model"]
feature_columns = artifact["feature_columns"]
threshold = artifact["threshold"]

# Load test data
features = pd.read_csv(
    "data/features.csv",
    parse_dates=["event_time"]
)
labels = pd.read_csv("data/labels.csv")

data = features.merge(
    labels[["event_id", "risk_label"]],
    on="event_id",
    validate="one_to_one"
)

data = data.sort_values(
    ["event_time", "event_id"]
).reset_index(drop=True)

test = data.iloc[int(len(data) * 0.80):].copy()

# Predict probabilities
test["risk_score"] = model.predict_proba(
    test[feature_columns]
)[:, 1]

test["prediction"] = (
    test["risk_score"] >= threshold
).astype(int)

print("\n=== HIGHEST RISK SCORES ===")

columns = [
    "event_id",
    "actor_id",
    "log_size_mb",
    "is_high_sensitivity",
    "is_unapproved_destination",
    "risk_label",
    "risk_score",
    "prediction",
]

print(
    test.sort_values(
        "risk_score",
        ascending=False
    )[columns].head(10).to_string(index=False)
)

print("\n=== MISCLASSIFIED EVENTS ===")

mistakes = test[
    test["risk_label"] != test["prediction"]
]

if mistakes.empty:
    print("No misclassified events in this test set.")
else:
    print(mistakes[columns].to_string(index=False))

print("\n=== MODEL COEFFICIENTS ===")

classifier = model.named_steps["classifier"]

coef = pd.DataFrame({
    "feature": feature_columns,
    "coefficient": classifier.coef_[0],
})

print(
    coef.sort_values(
        "coefficient",
        ascending=False
    ).to_string(index=False)
)

print("\nSelected threshold:", threshold)
