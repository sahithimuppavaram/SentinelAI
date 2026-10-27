
from pathlib import Path
import json

import numpy as np
import pandas as pd


SEED = 42
NUM_EMPLOYEES = 100
NUM_AGENTS = 20
NUM_DAYS = 30
NUM_INCIDENTS = 20

rng = np.random.default_rng(SEED)

OUTPUT = Path("data/v3")
OUTPUT.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Timestamp("2026-09-01", tz="UTC")

ROLES = {
    "software_engineer": {
        "avg_size": 25,
        "resources": ["code", "engineering_docs"],
    },
    "data_analyst": {
        "avg_size": 90,
        "resources": ["analytics", "reports"],
    },
    "finance_analyst": {
        "avg_size": 15,
        "resources": ["finance", "reports"],
    },
}

ALL_RESOURCES = [
    "code", "engineering_docs", "analytics",
    "reports", "finance", "restricted_records"
]

APPROVED_DESTINATIONS = ["internal", "approved_cloud"]

actors = []
events = []
event_labels = []
incidents = []

event_counter = 0


def add_event(
    actor,
    event_time,
    action,
    size_mb,
    resource_category,
    destination,
    workflow_id,
    incident_id=None,
):
    """Create an observable event and a separate label."""

    global event_counter
    event_counter += 1

    event_id = f"EV{event_counter:08d}"

    is_agent = actor["actor_type"] == "ai_agent"

    # Derive authorization using actor permissions.
    # This is not derived from the hidden risk label.
    allowed_resources = actor["allowed_resources"]
    allowed_destinations = actor["allowed_destinations"]

    authorized = (
        resource_category in allowed_resources
        and destination in allowed_destinations
        and (not is_agent or action == "tool_call")
    )

    sensitivity = (
        "restricted"
        if resource_category == "restricted_records"
        else "confidential"
        if resource_category in ["finance", "code"]
        else "internal"
    )

    events.append({
        "event_id": event_id,
        "event_time": event_time.isoformat(),
        "actor_id": actor["actor_id"],
        "actor_type": actor["actor_type"],
        "owner_employee_id": actor["owner_employee_id"],
        "session_id": workflow_id,
        "workflow_id": workflow_id,
        "action": action,
        "resource_id": (
            f"DOC{int(rng.integers(1, 10000)):05d}"
        ),
        "resource_category": resource_category,
        "sensitivity": sensitivity,
        "destination_type": destination,
        "size_mb": round(float(size_mb), 2),
        "source_type": (
            "agent_tool" if is_agent else "endpoint"
        ),
        "tool_name": (
            "document_access" if is_agent else None
        ),
        "authorization_status": (
            "authorized" if authorized else "unauthorized"
        ),
        "actor_timezone": actor["timezone"],
    })

    event_labels.append({
        "event_id": event_id,
        "risk_label": int(incident_id is not None),
        "incident_id": incident_id,
        "scenario_type": (
            "risk_worthy" if incident_id else "routine"
        ),
        "label_source": "synthetic_simulation",
        "label_available_at": (
            event_time + pd.Timedelta(days=2)
        ).isoformat(),
    })

    return event_id


# -------------------------------------------------
# 1. Generate employee profiles
# -------------------------------------------------

for i in range(1, NUM_EMPLOYEES + 1):
    role = str(rng.choice(list(ROLES.keys())))

    actors.append({
        "actor_id": f"E{i:03d}",
        "actor_type": "human",
        "owner_employee_id": None,
        "role": role,
        "timezone": "America/New_York",
        "allowed_resources": ROLES[role]["resources"],
        "allowed_destinations": APPROVED_DESTINATIONS,
        "avg_size_mb": ROLES[role]["avg_size"],
    })


# -------------------------------------------------
# 2. Generate AI-agent profiles
# -------------------------------------------------

for i in range(1, NUM_AGENTS + 1):
    owner = actors[i - 1]

    actors.append({
        "actor_id": f"A{i:03d}",
        "actor_type": "ai_agent",
        "owner_employee_id": owner["actor_id"],
        "role": f"{owner['role']}_agent",
        "timezone": owner["timezone"],
        "allowed_resources": owner["allowed_resources"],
        "allowed_destinations": APPROVED_DESTINATIONS,
        "avg_size_mb": owner["avg_size_mb"],
    })


# -------------------------------------------------
# 3. Generate routine activity histories
# -------------------------------------------------

for actor in actors:

    for day in range(NUM_DAYS):

        # Each actor gets several routine activities per day.
        daily_count = int(rng.integers(2, 6))

        for j in range(daily_count):

            hour = int(rng.integers(13, 23))
            minute = int(rng.integers(0, 60))

            event_time = START_DATE + pd.Timedelta(
                days=day,
                hours=hour,
                minutes=minute,
            )

            # Most events follow actor permissions.
            # Some unusual but legitimate work is included.
            unusual = bool(rng.random() < 0.04)

            size = rng.lognormal(
                mean=np.log(actor["avg_size_mb"]),
                sigma=0.8,
            )

            if unusual:
                size *= float(rng.uniform(3, 8))

            resource = str(
                rng.choice(actor["allowed_resources"])
            )

            destination = str(
                rng.choice(APPROVED_DESTINATIONS)
            )

            action = (
                "tool_call"
                if actor["actor_type"] == "ai_agent"
                else str(rng.choice([
                    "file_read",
                    "file_download",
                    "file_upload",
                ]))
            )

            add_event(
                actor=actor,
                event_time=event_time,
                action=action,
                size_mb=size,
                resource_category=resource,
                destination=destination,
                workflow_id=f"W-{actor['actor_id']}-{day:02d}-{j}",
            )


# -------------------------------------------------
# 4. Inject multi-event risk scenarios
# -------------------------------------------------

for i in range(1, NUM_INCIDENTS + 1):

    actor = actors[int(rng.integers(0, len(actors)))]

    incident_id = f"INC{i:04d}"
    workflow_id = f"W-{incident_id}"

    incident_start = START_DATE + pd.Timedelta(
        days=int(rng.integers(2, NUM_DAYS)),
        hours=int(rng.integers(13, 22)),
    )

    incident_event_ids = []

    if actor["actor_type"] == "ai_agent":
        actions = ["tool_call", "tool_call", "tool_call"]
    else:
        actions = [
            "file_read",
            "file_download",
            "file_upload",
        ]

    # A linked, three-event activity sequence
    for step, action in enumerate(actions):

        event_time = incident_start + pd.Timedelta(
            minutes=step * 4
        )

        # Include smaller and larger risk-worthy transfers.
        size = float(rng.lognormal(
            mean=np.log(30),
            sigma=1.2,
        ))

        destination = (
            "unapproved_external"
            if step == 2
            else "internal"
        )

        event_id = add_event(
            actor=actor,
            event_time=event_time,
            action=action,
            size_mb=size,
            resource_category="restricted_records",
            destination=destination,
            workflow_id=workflow_id,
            incident_id=incident_id,
        )

        incident_event_ids.append(event_id)

    incidents.append({
        "incident_id": incident_id,
        "actor_id": actor["actor_id"],
        "start_time": incident_start.isoformat(),
        "scenario_type": (
            "ai_agent_permission_violation"
            if actor["actor_type"] == "ai_agent"
            else "sensitive_data_access_sequence"
        ),
        "event_count": len(incident_event_ids),
    })


# -------------------------------------------------
# 5. Export separate datasets
# -------------------------------------------------

actors_df = pd.DataFrame(actors)

# Serialize list-valued actor permissions for CSV.
for column in ["allowed_resources", "allowed_destinations"]:
    actors_df[column] = actors_df[column].apply(
        json.dumps
    )

events_df = pd.DataFrame(events).sort_values(
    ["event_time", "event_id"]
)

labels_df = pd.DataFrame(event_labels)

incidents_df = pd.DataFrame(incidents)

actors_df.to_csv(
    OUTPUT / "actors.csv", index=False
)

events_df.to_csv(
    OUTPUT / "events.csv", index=False
)

labels_df.to_csv(
    OUTPUT / "event_labels.csv", index=False
)

incidents_df.to_csv(
    OUTPUT / "incidents.csv", index=False
)

metadata = {
    "version": "v3",
    "seed": SEED,
    "employees": NUM_EMPLOYEES,
    "agents": NUM_AGENTS,
    "days": NUM_DAYS,
    "total_events": len(events_df),
    "risk_events": int(labels_df["risk_label"].sum()),
    "incidents": len(incidents_df),
    "note": "Synthetic educational data; not validated security data",
}

with open(OUTPUT / "simulation_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)


print("\n=== V3 SIMULATION COMPLETE ===")
print("Total actors:", len(actors_df))
print("Total events:", len(events_df))
print("Risk-worthy events:", metadata["risk_events"])
print("Incidents:", len(incidents_df))

print("\nGenerated files:")
for file in OUTPUT.iterdir():
    print("-", file)
