
from pathlib import Path
import json

import pandas as pd


# =========================================
# CONFIGURATION
# =========================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/v3_2"
OUTPUT_DIR = DATA_DIR / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CUTOFF = pd.Timestamp(
    "2026-09-19", tz="UTC"
)

VALIDATION_CUTOFF = pd.Timestamp(
    "2026-09-25", tz="UTC"
)


# =========================================
# 1. LOAD FEATURES AND LABELS
# =========================================

features = pd.read_csv(
    DATA_DIR / "temporal_features.csv"
)

labels = pd.read_csv(
    DATA_DIR / "event_labels.csv"
)

assert features["event_id"].is_unique
assert labels["event_id"].is_unique

assert set(features["event_id"]) == set(
    labels["event_id"]
)

# Keep labels in the evaluation dataset,
# but never include them in model inputs.
data = features.merge(
    labels[
        [
            "event_id",
            "risk_label",
            "incident_id",
            "scenario_type",
            "label_available_at",
        ]
    ],
    on="event_id",
    validate="one_to_one",
)

data["event_time"] = pd.to_datetime(
    data["event_time"],
    utc=True,
    errors="coerce",
)

data["label_available_at"] = pd.to_datetime(
    data["label_available_at"],
    utc=True,
    errors="coerce",
)

assert data["event_time"].notna().all()
assert data["label_available_at"].notna().all()

assert (
    data["label_available_at"]
    >= data["event_time"]
).all()

assert data["workflow_id"].notna().all()


# =========================================
# 2. COMPUTE WORKFLOW TIME BOUNDARIES
# =========================================

workflows = data.groupby(
    "workflow_id", sort=False
).agg(
    first_event=("event_time", "min"),
    last_event=("event_time", "max"),
    last_label_available=(
        "label_available_at", "max"
    ),
)


def assign_workflow(row):
    """
    Keep each workflow in exactly one
    partition or exclude it entirely.

    No partially assigned workflows.
    """

    first = row["first_event"]
    last = row["last_event"]
    label_ready = row["last_label_available"]

    # Training requires the complete workflow
    # and all its labels to be mature.
    if last < TRAIN_CUTOFF:
        if label_ready <= TRAIN_CUTOFF:
            return "train"
        return "excluded"

    # Validation begins at the training cutoff.
    if (
        first >= TRAIN_CUTOFF
        and last < VALIDATION_CUTOFF
    ):
        return "validation"

    # Test begins at the validation cutoff.
    if first >= VALIDATION_CUTOFF:
        return "test"

    # Workflows spanning partition boundaries
    # are excluded, not split apart.
    return "excluded"


workflows["partition"] = workflows.apply(
    assign_workflow,
    axis=1,
)

data["partition"] = (
    data["workflow_id"].map(
        workflows["partition"]
    )
)

assert data["partition"].notna().all()


# =========================================
# 3. CHECK INCIDENT ISOLATION
# =========================================

incident_partition_counts = (
    data.dropna(subset=["incident_id"])
    .groupby("incident_id")["partition"]
    .nunique()
)

assert (
    incident_partition_counts <= 1
).all(), "Incident crosses partitions"


workflow_partition_counts = (
    data.groupby("workflow_id")["partition"]
    .nunique()
)

assert (
    workflow_partition_counts == 1
).all(), "Workflow crosses partitions"


# =========================================
# 4. CREATE PARTITIONS
# =========================================

partitions = {}

for name in [
    "train",
    "validation",
    "test",
    "excluded",
]:
    subset = data[
        data["partition"] == name
    ].copy()

    subset = subset.sort_values(
        ["event_time", "event_id"]
    )

    subset.to_csv(
        OUTPUT_DIR / f"{name}.csv",
        index=False,
    )

    partitions[name] = subset


# =========================================
# 5. VERIFY DATE AND LABEL RULES
# =========================================

train = partitions["train"]
validation = partitions["validation"]
test = partitions["test"]

assert (
    train["event_time"] < TRAIN_CUTOFF
).all()

assert (
    train["label_available_at"]
    <= TRAIN_CUTOFF
).all()

assert (
    validation["event_time"]
    >= TRAIN_CUTOFF
).all()

assert (
    validation["event_time"]
    < VALIDATION_CUTOFF
).all()

assert (
    test["event_time"]
    >= VALIDATION_CUTOFF
).all()

# All events are accounted for.
total = sum(
    len(df) for df in partitions.values()
)

assert total == len(data)

all_saved_ids = []

for df in partitions.values():
    all_saved_ids.extend(
        df["event_id"].tolist()
    )

assert len(all_saved_ids) == len(
    set(all_saved_ids)
)

assert set(all_saved_ids) == set(
    data["event_id"]
)


# =========================================
# 6. REPORT DISTRIBUTION
# =========================================

print("\n=== V3.2 CHRONOLOGICAL SPLITS ===")

summary = {}

for name, df in partitions.items():
    total_events = len(df)
    risky_events = int(
        df["risk_label"].sum()
    )

    incident_count = (
        df["incident_id"]
        .dropna()
        .nunique()
    )

    workflow_count = (
        df["workflow_id"]
        .nunique()
    )

    print(
        f"{name}: "
        f"{total_events:,} events, "
        f"{risky_events} risky, "
        f"{incident_count} incidents, "
        f"{workflow_count} workflows"
    )

    print("  Scenarios:")
    print(
        df["scenario_type"]
        .value_counts()
        .to_string()
    )

    summary[name] = {
        "events": total_events,
        "risk_events": risky_events,
        "incidents": int(incident_count),
        "workflows": int(workflow_count),
    }


# =========================================
# 7. SAVE METADATA
# =========================================

metadata = {
    "dataset": "v3.2",
    "train_cutoff": TRAIN_CUTOFF.isoformat(),
    "validation_cutoff": (
        VALIDATION_CUTOFF.isoformat()
    ),
    "strategy": (
        "Chronological workflow-level "
        "splitting with delayed training "
        "label availability"
    ),
    "partitions": summary,
}

with open(
    OUTPUT_DIR / "split_metadata.json",
    "w",
) as file:
    json.dump(
        metadata,
        file,
        indent=2,
    )

print("\nPASS: Workflows do not cross partitions")
print("PASS: Incidents do not cross partitions")
print("PASS: Training labels are mature")
print("PASS: Every event is accounted for")
print("\nSaved to:", OUTPUT_DIR)
