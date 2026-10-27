
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/v3")
SPLIT_DIR = DATA_DIR / "splits"

events = pd.read_csv(DATA_DIR / "events.csv")
labels = pd.read_csv(DATA_DIR / "event_labels.csv")

train = pd.read_csv(SPLIT_DIR / "train.csv")
val = pd.read_csv(SPLIT_DIR / "validation.csv")
test = pd.read_csv(SPLIT_DIR / "test.csv")

all_ids = set(events["event_id"])

assigned_ids = (
    set(train["event_id"])
    | set(val["event_id"])
    | set(test["event_id"])
)

excluded_ids = all_ids - assigned_ids

# Build an audit table for excluded events
excluded = events[
    events["event_id"].isin(excluded_ids)
].merge(
    labels[
        ["event_id", "risk_label",
         "incident_id", "label_available_at"]
    ],
    on="event_id",
    validate="one_to_one"
)

print("\n=== SPLIT AUDIT ===")

print("Total events:", len(events))
print("Assigned events:", len(assigned_ids))
print("Excluded events:", len(excluded_ids))

print("\n=== RISKY EVENTS BY SPLIT ===")

for name, df in [
    ("train", train),
    ("validation", val),
    ("test", test),
    ("excluded", excluded),
]:
    incidents = df["incident_id"].dropna().nunique()

    print(
        f"{name}: "
        f"{len(df)} events, "
        f"{int(df['risk_label'].sum())} risky, "
        f"{incidents} incidents"
    )

print("\n=== EXCLUDED EVENT PERIOD ===")

if not excluded.empty:
    times = pd.to_datetime(
        excluded["event_time"], utc=True
    )

    print("Earliest:", times.min())
    print("Latest:", times.max())

    print("\nExcluded risk labels:")
    print(excluded["risk_label"].value_counts())

# Verify complete partition accounting
assert len(assigned_ids) + len(excluded_ids) == len(all_ids)

# Verify split IDs do not overlap
assert set(train["event_id"]).isdisjoint(
    set(val["event_id"])
)
assert set(train["event_id"]).isdisjoint(
    set(test["event_id"])
)
assert set(val["event_id"]).isdisjoint(
    set(test["event_id"])
)

# Verify incident membership does not cross any group
groups = {
    "train": train,
    "validation": val,
    "test": test,
    "excluded": excluded,
}

for name_a, df_a in groups.items():
    for name_b, df_b in groups.items():
        if name_a >= name_b:
            continue

        ids_a = set(df_a["incident_id"].dropna())
        ids_b = set(df_b["incident_id"].dropna())

        assert ids_a.isdisjoint(ids_b), (
            f"Incident leakage: {name_a} vs {name_b}"
        )

print("\nPASS: Split audit completed")
