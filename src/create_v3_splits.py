
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/v3")
OUTPUT = DATA_DIR / "splits"
OUTPUT.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(
    DATA_DIR / "contextual_features.csv"
)

labels = pd.read_csv(
    DATA_DIR / "event_labels.csv"
)

data = features.merge(
    labels[
        ["event_id", "risk_label",
         "incident_id", "label_available_at"]
    ],
    on="event_id",
    validate="one_to_one"
)

data["event_time"] = pd.to_datetime(
    data["event_time"], utc=True
)

data["label_available_at"] = pd.to_datetime(
    data["label_available_at"], utc=True
)

# Fixed date boundaries for September simulation
train_cutoff = pd.Timestamp("2026-09-19", tz="UTC")
val_cutoff = pd.Timestamp("2026-09-25", tz="UTC")

# Keep only mature labels for training.
# This leaves a two-day maturity gap before train cutoff.
train = data[
    (data["event_time"] < train_cutoff)
    & (data["label_available_at"] <= train_cutoff)
].copy()

validation = data[
    (data["event_time"] >= train_cutoff)
    & (data["event_time"] < val_cutoff)
].copy()

test = data[
    data["event_time"] >= val_cutoff
].copy()

# Check incident membership across partitions
def incident_ids(df):
    return set(df["incident_id"].dropna())

train_incidents = incident_ids(train)
val_incidents = incident_ids(validation)
test_incidents = incident_ids(test)

assert train_incidents.isdisjoint(val_incidents)
assert train_incidents.isdisjoint(test_incidents)
assert val_incidents.isdisjoint(test_incidents)

# Check event IDs are unique across partitions
assert set(train["event_id"]).isdisjoint(
    set(validation["event_id"])
)

assert set(train["event_id"]).isdisjoint(
    set(test["event_id"])
)

assert set(validation["event_id"]).isdisjoint(
    set(test["event_id"])
)

# Ensure training labels were available
assert (
    train["label_available_at"] <= train_cutoff
).all()

# Save splits
for name, dataset in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    dataset.sort_values(
        ["event_time", "event_id"]
    ).to_csv(
        OUTPUT / f"{name}.csv",
        index=False
    )

    print(
        f"{name}: {len(dataset):,} rows, "
        f"{int(dataset['risk_label'].sum())} risky"
    )

print("\nPASS: Chronological incident-aware splits created")
print("Saved to:", OUTPUT)
