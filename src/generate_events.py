
from pathlib import Path

import numpy as np
import pandas as pd


# Reproducible random number generator
rng = np.random.default_rng(42)

# Define our simulated users
employees = [f"E{i:03d}" for i in range(1, 51)]
agents = [f"A{i:03d}" for i in range(1, 11)]
actors = employees + agents

# Every AI agent belongs to an employee
agent_owners = {
    agent: employees[i]
    for i, agent in enumerate(agents)
}

events = []
labels = []

start_date = pd.Timestamp("2026-09-01", tz="UTC")

for i in range(1000):
    actor_id = str(rng.choice(actors))
    is_agent = actor_id.startswith("A")

    # Hidden synthetic scenario
    is_risky = bool(rng.random() < 0.03)

    # Some risky events occur during normal work hours,
    # and some routine events occur outside those hours.
    off_hours = bool(
        rng.random() < (0.70 if is_risky else 0.10)
    )

    hour = int(
        rng.integers(0, 7)
        if off_hours
        else rng.integers(8, 19)
    )

    event_time = start_date + pd.Timedelta(
        days=int(rng.integers(0, 30)),
        hours=hour,
        minutes=int(rng.integers(0, 60))
    )

    if is_agent:
        action = "tool_call"
    elif is_risky:
        action = str(rng.choice(
            ["file_download", "file_upload"]
        ))
    else:
        action = str(rng.choice(
            ["file_read", "file_download", "file_upload"]
        ))

    size_mb = round(float(
        rng.uniform(100, 750)
        if is_risky
        else rng.lognormal(2, 1)
    ), 2)

    destination = str(rng.choice(
        ["internal", "approved_cloud", "unapproved_external"],
        p=[0.10, 0.20, 0.70] if is_risky
        else [0.65, 0.30, 0.05]
    ))

    sensitivity = str(rng.choice(
        ["public", "internal", "confidential", "restricted"],
        p=[0.05, 0.15, 0.45, 0.35] if is_risky
        else [0.25, 0.45, 0.25, 0.05]
    ))

    event_id = f"EV{i+1:06d}"

    events.append({
        "event_id": event_id,
        "event_time": event_time.isoformat(),
        "actor_id": actor_id,
        "actor_type": "ai_agent" if is_agent else "human",
        "owner_employee_id": (
            agent_owners[actor_id] if is_agent else None
        ),
        "session_id": f"S{int(rng.integers(1, 500)):04d}",
        "action": action,
        "resource_id": f"DOC{int(rng.integers(1, 500)):05d}",
        "sensitivity": sensitivity,
        "destination_type": destination,
        "size_mb": size_mb,
        "source_type": (
            "agent_tool" if is_agent else "endpoint"
        ),
    })

    labels.append({
        "event_id": event_id,
        "risk_label": int(is_risky),
        "scenario_type": (
            "simulated_risk" if is_risky else "routine"
        ),
        "label_source": "synthetic_simulation",
        "label_available_at": (
            event_time + pd.Timedelta(days=2)
        ).isoformat(),
    })


# Save events and labels separately
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

pd.DataFrame(events).sort_values(
    "event_time"
).to_csv(output_dir / "events.csv", index=False)

pd.DataFrame(labels).to_csv(
    output_dir / "labels.csv", index=False
)

print("Successfully generated 1,000 activity events!")
print("Files created: data/events.csv and data/labels.csv")
