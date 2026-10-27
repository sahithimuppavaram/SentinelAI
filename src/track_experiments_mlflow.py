
from pathlib import Path
import json

import joblib
import mlflow
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/v3_2"
SPLIT_DIR = ROOT / "data/v3_2/splits"
TRACKING_DIR = ROOT / "mlruns"

TRACKING_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(
    f"sqlite:///{ROOT / 'mlflow.db'}"
)
mlflow.set_experiment("SentinelAI-V3.2")

with open(
    MODEL_DIR / "comparison.json"
) as f:
    supervised = json.load(f)

with open(
    MODEL_DIR / "anomaly_comparison.json"
) as f:
    anomaly = json.load(f)

train = pd.read_csv(SPLIT_DIR / "train.csv")
validation = pd.read_csv(
    SPLIT_DIR / "validation.csv"
)
test = pd.read_csv(SPLIT_DIR / "test.csv")

selected_model = supervised["selected_model"]


def log_experiment(
    name,
    artifact_filename,
    validation_metrics,
    test_metrics=None,
):
    artifact_path = MODEL_DIR / artifact_filename
    artifact = joblib.load(artifact_path)

    with mlflow.start_run(run_name=name) as run:

        mlflow.log_params({
            "dataset_version": "v3.2",
            "algorithm": name,
            "training_events": len(train),
            "validation_events": len(validation),
            "test_events": len(test),
            "feature_count": len(
                artifact["feature_columns"]
            ),
            "alert_target_per_1000": 20,
        })

        for key, value in validation_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(
                    f"validation_{key}",
                    float(value),
                )

        if test_metrics is not None:
            for key, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(
                        f"test_{key}",
                        float(value),
                    )

        mlflow.set_tags({
            "project": "SentinelAI",
            "data_source": "synthetic",
            "selected_supervised_model": str(
                name == selected_model
            ).lower(),
        })

        mlflow.log_artifact(str(artifact_path))

        print(
            f"PASS: Logged {name} | "
            f"Run ID: {run.info.run_id}"
        )


# Supervised model experiments
for name in [
    "logistic_regression",
    "xgboost",
]:
    results = supervised[name]

    # Only the validation-selected model
    # received a final supervised test evaluation.
    test_metrics = (
        supervised["selected_test_metrics"]
        if name == selected_model
        else None
    )

    log_experiment(
        name=name,
        artifact_filename=f"{name}.joblib",
        validation_metrics=results["validation"],
        test_metrics=test_metrics,
    )


# Unsupervised anomaly detection experiment
log_experiment(
    name="isolation_forest",
    artifact_filename="isolation_forest.joblib",
    validation_metrics=anomaly["validation"],
    test_metrics=anomaly["test"],
)

print("\n=== MLFLOW TRACKING COMPLETE ===")
print("Experiment: SentinelAI-V3.2")
print("Tracked models: 3")
print("Tracking directory:", TRACKING_DIR)

