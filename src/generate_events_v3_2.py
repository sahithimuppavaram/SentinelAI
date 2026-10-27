
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

SEED = 84

SOURCE = Path("data/v3")
OUTPUT = Path("data/v3_2")
OUTPUT.mkdir(parents=True, exist_ok=True)

NUM_RISK_WORKFLOWS = 120
NUM_LEGITIMATE_WORKFLOWS = 120

START_DATE = pd.Timestamp(
    "2026-09-01", tz="UTC"
)

rng = np.random.default_rng(SEED)


# ==================================================
# 1. LOAD EXISTING V3.1 DATA
# ==================================================

actors = pd.read_csv(SOURCE / "actors.csv")
events = pd.read_csv(SOURCE / "events.csv")
labels = pd.read_csv(SOURCE / "event_labels.csv")
incidents = pd.read_csv(SOURCE / "incidents.csv")

# Convert permission strings back to Python lists
actors["allowed_resources"] = actors[
    "allowed_resources"
].apply(json.loads)

actors["allowed_destinations"] = actors[
    "allowed_destinations"
].apply(json.loads)

actor_records = actors.to_dict("records")

new_events = []
new_labels = []
new_incidents = []

next_event_number = (
    events["event_id"]
    .str.replace("EV", "", regex=False)
    .astype(int)
    .max()
    + 1
)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def create_event(
    actor,
    event_time,
    action,
    resource,
    resource_id,
    destination,
    size_mb,
    workflow_id,
    risk_label,
    scenario_type,
    incident_id=None,
):
    global next_event_number

    event_id = f"EV{next_event_number:08d}"
    next_event_number += 1

    is_agent = actor["actor_type"] == "ai_agent"

    authorized = (
        resource in actor["allowed_resources"]
        and destination in actor["allowed_destinations"]
        and (not is_agent or action == "tool_call")
    )

    sensitivity = (
        "confidential"
        if resource in ["code", "finance"]
        else "internal"
    )

    new_events.append({
        "event_id": event_id,
        "event_time": event_time.isoformat(),
        "actor_id": actor["actor_id"],
        "actor_type": actor["actor_type"],
        "owner_employee_id": actor["owner_employee_id"],
        "session_id": workflow_id,
        "workflow_id": workflow_id,
        "action": action,
        "resource_id": resource_id,
        "resource_category": resource,
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
            "authorized" if authorized
            else "unauthorized"
        ),
        "actor_timezone": actor["timezone"],
    })

    new_labels.append({
        "event_id": event_id,
        "risk_label": risk_label,
        "incident_id": incident_id,
        "scenario_type": scenario_type,
        "label_source": "synthetic_simulation",
        "label_available_at": (
            event_time + pd.Timedelta(days=2)
        ).isoformat(),
    })

    return event_id


def create_workflow(workflow_number, risky):
    """
    Generate a connected multi-event workflow.

    Both classes use approved resources and
    destinations with overlapping distributions.
    """

    actor = actor_records[
        int(rng.integers(0, len(actor_records)))
    ]

    workflow_id = f"V32-W{workflow_number:04d}"

    scenario_type = (
        "authorized_bulk_misuse"
        if risky
        else "approved_bulk_operation"
    )

    incident_id = (
        f"V32-INC{workflow_number:04d}"
        if risky
        else None
    )

    day = int(rng.integers(1, 30))
    hour = int(rng.integers(13, 22))
    minute = int(rng.integers(0, 40))

    current_time = START_DATE + pd.Timedelta(
        days=day,
        hours=hour,
        minutes=minute,
    )

    start_time = current_time

    # Both classes contain variable-length workflows.
    event_count = int(rng.integers(3, 7))

    resource = str(
        rng.choice(actor["allowed_resources"])
    )

    destination = str(
        rng.choice(actor["allowed_destinations"])
    )

    event_ids = []

    # Similar, but not identical, size distributions
    # for risky and legitimate workflows.
    multiplier = (
        float(rng.uniform(2.5, 6.0))
        if risky
        else float(rng.uniform(1.8, 5.5))
    )

    # Select document identifiers from a workflow pool.
    # Both classes can revisit the same documents.
    document_pool = [
        f"DOC{int(rng.integers(1, 10000)):05d}"
        for _ in range(
            max(2, event_count // 2)
        )
    ]

    for step in range(event_count):

        current_time += pd.Timedelta(
            minutes=int(rng.integers(1, 8))
        )

        if actor["actor_type"] == "ai_agent":
            action = "tool_call"
        else:
            action = (
                "file_upload"
                if step == event_count - 1
                else str(rng.choice([
                    "file_read",
                    "file_download",
                ]))
            )

        size_mb = float(rng.lognormal(
            mean=np.log(actor["avg_size_mb"]),
            sigma=0.7,
        )) * multiplier

        # Some legitimate workflows are larger than
        # risky workflows, creating harder negatives.
        if not risky and rng.random() < 0.25:
            size_mb *= 2.0

        resource_id = str(
            rng.choice(document_pool)
        )

        event_id = create_event(
            actor=actor,
            event_time=current_time,
            action=action,
            resource=resource,
            resource_id=resource_id,
            destination=destination,
            size_mb=size_mb,
            workflow_id=workflow_id,
            risk_label=int(risky),
            scenario_type=scenario_type,
            incident_id=incident_id,
        )

        event_ids.append(event_id)

    if risky:
        new_incidents.append({
            "incident_id": incident_id,
            "actor_id": actor["actor_id"],
            "start_time": start_time.isoformat(),
            "scenario_type": scenario_type,
            "event_count": len(event_ids),
        })


# ==================================================
# 3. GENERATE ADDITIONAL WORKFLOWS
# ==================================================

for i in range(1, NUM_RISK_WORKFLOWS + 1):
    create_workflow(i, risky=True)

for i in range(
    NUM_RISK_WORKFLOWS + 1,
    NUM_RISK_WORKFLOWS + NUM_LEGITIMATE_WORKFLOWS + 1,
):
    create_workflow(i, risky=False)


# ==================================================
# 4. COMBINE OLD AND NEW DATA
# ==================================================

events_v32 = pd.concat(
    [events, pd.DataFrame(new_events)],
    ignore_index=True,
)

labels_v32 = pd.concat(
    [labels, pd.DataFrame(new_labels)],
    ignore_index=True,
)

incidents_v32 = pd.concat(
    [incidents, pd.DataFrame(new_incidents)],
    ignore_index=True,
)

events_v32["event_time"] = pd.to_datetime(
    events_v32["event_time"],
    utc=True,
)

events_v32 = events_v32.sort_values(
    ["event_time", "event_id"]
)

# Re-serialize actor permissions for CSV.
actors_export = actors.copy()

for column in [
    "allowed_resources",
    "allowed_destinations",
]:
    actors_export[column] = (
        actors_export[column].apply(json.dumps)
    )


# ==================================================
# 5. VALIDATION
# ==================================================

assert events_v32["event_id"].is_unique
assert labels_v32["event_id"].is_unique

assert set(events_v32["event_id"]) == set(
    labels_v32["event_id"]
)

assert incidents_v32["incident_id"].is_unique

assert labels_v32["risk_label"].isin([0, 1]).all()

new_events_df = pd.DataFrame(new_events)
new_labels_df = pd.DataFrame(new_labels)

assert (
    new_events_df["authorization_status"]
    == "authorized"
).all()

assert (
    new_labels_df["risk_label"].sum() > 0
)

assert (
    (new_labels_df["risk_label"] == 0).sum() > 0
)


# ==================================================
# 6. EXPORT V3.2 DATA
# ==================================================

actors_export.to_csv(
    OUTPUT / "actors.csv", index=False
)

events_v32.to_csv(
    OUTPUT / "events.csv", index=False
)

labels_v32.to_csv(
    OUTPUT / "event_labels.csv", index=False
)

incidents_v32.to_csv(
    OUTPUT / "incidents.csv", index=False
)

metadata = {
    "version": "v3.2",
    "seed": SEED,
    "base_version": "v3.1",
    "total_events": len(events_v32),
    "risk_events": int(
        labels_v32["risk_label"].sum()
    ),
    "incidents": len(incidents_v32),
    "new_risky_workflows": NUM_RISK_WORKFLOWS,
    "new_legitimate_workflows": (
        NUM_LEGITIMATE_WORKFLOWS
    ),
    "note": (
        "Synthetic educational dataset. "
        "Scenario labels do not imply "
        "verified malicious behavior."
    ),
}

with open(
    OUTPUT / "simulation_metadata.json",
    "w",
) as f:
    json.dump(metadata, f, indent=2)


# ==================================================
# 7. REPORT
# ==================================================

print("\n=== V3.2 SIMULATION COMPLETE ===")

print("Total events:", len(events_v32))
print(
    "Risk-worthy events:",
    int(labels_v32["risk_label"].sum())
)
print("Incidents:", len(incidents_v32))

print("\nNew workflow scenario distribution:")
print(
    new_labels_df[
        "scenario_type"
    ].value_counts().to_string()
)

print("\nNew event authorization vs risk:")
print(
    pd.crosstab(
        new_events_df["authorization_status"],
        new_labels_df["risk_label"],
    )
)

print("\nPASS: V3.2 dataset generated")
print("Saved to:", OUTPUT)
