
import pandas as pd

# Load datasets
events = pd.read_csv("data/events.csv")
labels = pd.read_csv("data/labels.csv")

# Convert timestamp fields
events["event_time"] = pd.to_datetime(
    events["event_time"], utc=True, errors="coerce"
)

labels["label_available_at"] = pd.to_datetime(
    labels["label_available_at"], utc=True, errors="coerce"
)

errors = []

# 1. Check unique identifiers
if events["event_id"].isna().any():
    errors.append("Missing event IDs")

if events["event_id"].duplicated().any():
    errors.append("Duplicate event IDs")

# 2. Check timestamps
if events["event_time"].isna().any():
    errors.append("Invalid event timestamps")

if labels["label_available_at"].isna().any():
    errors.append("Invalid label timestamps")

# 3. Check transfer sizes
if events["size_mb"].isna().any():
    errors.append("Missing transfer sizes")

if (events["size_mb"] < 0).any():
    errors.append("Negative transfer sizes")

# 4. Check actor types
valid_actor_types = {"human", "ai_agent"}

if not events["actor_type"].isin(valid_actor_types).all():
    errors.append("Invalid actor types")

# 5. Check AI-agent ownership
agents = events[events["actor_type"] == "ai_agent"]

if agents["owner_employee_id"].isna().any():
    errors.append("AI agents missing owners")

# 6. Check expected labels
if labels["event_id"].duplicated().any():
    errors.append("Duplicate label IDs")

if not labels["risk_label"].isin([0, 1]).all():
    errors.append("Invalid risk labels")

# 7. Check that labels match existing events
if set(labels["event_id"]) != set(events["event_id"]):
    errors.append("Event-label ID mismatch")

# 8. Check required event values
required_columns = [
    "actor_id",
    "session_id",
    "action",
    "resource_id",
    "sensitivity",
    "destination_type",
    "source_type",
]

for column in required_columns:
    if events[column].isna().any():
        errors.append(f"Missing required values: {column}")

# Report findings
print("\n=== DATA VALIDATION RESULTS ===")

if errors:
    for error in errors:
        print("FAIL:", error)
    raise SystemExit(1)
else:
    print("PASS: All validation checks passed")
    print(f"Validated {len(events):,} activity events")
    print(f"Validated {len(labels):,} ground-truth labels")
