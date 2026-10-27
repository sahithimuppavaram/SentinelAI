
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
NUM_EVENTS = 10000

rng = np.random.default_rng(SEED)

# Define employee profiles
profiles = {
    "software_engineer": {
        "avg_size_mb": 25,
        "sensitivity_prob": 0.20,
    },
    "data_analyst": {
        "avg_size_mb": 100,
        "sensitivity_prob": 0.35,
    },
    "finance_analyst": {
        "avg_size_mb": 15,
        "sensitivity_prob": 0.65,
    },
}

employees = {}

for i in range(1, 101):
    employee_id = f"E{i:03d}"
    role = str(rng.choice(list(profiles.keys())))
    employees[employee_id] = role

agents = {
    f"A{i:03d}": f"E{i:03d}"
    for i in range(1, 21)
}

events = []
labels = []

start_date = pd.Timestamp("2026-09-01", tz="UTC")

for i in range(NUM_EVENTS):

    actor_id = str(
        rng.choice(list(employees.keys()) + list(agents.keys()))
    )

    is_agent = actor_id in agents

    employee_id = agents[actor_id] if is_agent else actor_id
    role = employees[employee_id]
    profile = profiles[role]

    # Hidden scenario label
    scenario = str(rng.choice(
        ["routine", "legitimate_unusual", "risk_worthy"],
        p=[0.94, 0.04, 0.02]
    ))

    # Simulate behavior with overlapping patterns
    base_size = float(
        rng.lognormal(
            mean=np.log(profile["avg_size_mb"]),
            sigma=0.9
        )
    )

    if scenario == "legitimate_unusual":
        size_mb = base_size * float(rng.uniform(3, 8))
    elif scenario == "risk_worthy":
        size_mb = base_size * float(rng.uniform(0.5, 7))
    else:
        size_mb = base_size

    size_mb = round(size_mb, 2)

    # Assign destination probabilities by scenario
    destination_probs = {
        "routine": [0.65, 0.30, 0.05],
        "legitimate_unusual": [0.20, 0.70, 0.10],
        "risk_worthy": [0.20, 0.20, 0.60]
    }

    destination = str(rng.choice(
        ["internal", "approved_cloud", "unapproved_external"],
        p=destination_probs[scenario]
    ))

    # High sensitivity varies by employee role
    high_sensitivity = bool(
        rng.random() < profile["sensitivity_prob"]
    )

    sensitivity = (
        str(rng.choice(["confidential", "restricted"]))
        if high_sensitivity
        else str(rng.choice(["public", "internal"]))
    )

    # Off-hours occur in all scenarios
    off_hours_prob = {
        "routine": 0.10,
        "legitimate_unusual": 0.50,
        "risk_worthy": 0.65
    }

    off_hours = bool(
        rng.random() < off_hours_prob[scenario]
    )

    hour = int(
        rng.integers(0, 7)
        if off_hours
        else rng.integers(8, 19)
    )

    event_time = start_date + pd.Timedelta(
        days=int(rng.integers(0, 60)),
        hours=hour,
        minutes=int(rng.integers(0, 60))
    )

    if is_agent:
        action = "tool_call"
    else:
        action = str(rng.choice([
            "file_read",
            "file_download",
            "file_upload"
        ]))

    event_id = f"EV{i+1:07d}"

    events.append({
        "event_id": event_id,
        "event_time": event_time.isoformat(),
        "actor_id": actor_id,
        "actor_type": "ai_agent" if is_agent else "human",
        "owner_employee_id": (
            employee_id if is_agent else None
        ),
        "session_id": f"S{int(rng.integers(1, 2000)):05d}",
        "action": action,
        "resource_id": f"DOC{int(rng.integers(1, 3000)):05d}",
        "sensitivity": sensitivity,
        "destination_type": destination,
        "size_mb": size_mb,
        "source_type": (
            "agent_tool" if is_agent else "endpoint"
        ),
    })

    labels.append({
        "event_id": event_id,
        "risk_label": int(scenario == "risk_worthy"),
        "scenario_type": scenario,
        "label_source": "synthetic_simulation",
        "label_available_at": (
            event_time + pd.Timedelta(days=2)
        ).isoformat(),
    })


output_dir = Path("data/v2")
output_dir.mkdir(parents=True, exist_ok=True)

events_df = pd.DataFrame(events).sort_values("event_time")
labels_df = pd.DataFrame(labels)

events_df.to_csv(output_dir / "events.csv", index=False)
labels_df.to_csv(output_dir / "labels.csv", index=False)

print("\n=== V2 DATA GENERATION COMPLETE ===")
print("Total events:", len(events_df))
print("\nScenario counts:")
print(labels_df["scenario_type"].value_counts())
print("\nFiles:")
print(output_dir / "events.csv")
print(output_dir / "labels.csv")
