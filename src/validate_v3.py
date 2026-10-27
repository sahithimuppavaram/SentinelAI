
import json

import pandas as pd


DATA_DIR = "data/v3"

actors = pd.read_csv(f"{DATA_DIR}/actors.csv")
events = pd.read_csv(f"{DATA_DIR}/events.csv")
labels = pd.read_csv(f"{DATA_DIR}/event_labels.csv")
incidents = pd.read_csv(f"{DATA_DIR}/incidents.csv")

errors = []


def check(condition, message):
    if condition:
        print(f"PASS: {message}")
    else:
        print(f"FAIL: {message}")
        errors.append(message)


# 1. Unique identifiers
check(
    actors["actor_id"].is_unique,
    "Actor IDs are unique"
)

check(
    events["event_id"].is_unique,
    "Event IDs are unique"
)

check(
    labels["event_id"].is_unique,
    "Label IDs are unique"
)

check(
    incidents["incident_id"].is_unique,
    "Incident IDs are unique"
)


# 2. Actor references
check(
    events["actor_id"].isin(actors["actor_id"]).all(),
    "All events reference known actors"
)

# 3. AI-agent ownership
employees = set(
    actors.loc[
        actors["actor_type"] == "human", "actor_id"
    ]
)

agents = actors[actors["actor_type"] == "ai_agent"]

check(
    agents["owner_employee_id"].isin(employees).all(),
    "All AI agents have valid employee owners"
)


# 4. Event-label consistency
check(
    set(events["event_id"]) == set(labels["event_id"]),
    "Every event has exactly one label"
)

check(
    labels["risk_label"].isin([0, 1]).all(),
    "Risk labels are binary"
)


# 5. Validate timestamps
events["event_time"] = pd.to_datetime(
    events["event_time"],
    utc=True,
    errors="coerce"
)

labels["label_available_at"] = pd.to_datetime(
    labels["label_available_at"],
    utc=True,
    errors="coerce"
)

check(
    events["event_time"].notna().all(),
    "Event timestamps are valid"
)

check(
    labels["label_available_at"].notna().all(),
    "Label timestamps are valid"
)


# 6. Incident relationships
risk_labels = labels[labels["risk_label"] == 1]

check(
    risk_labels["incident_id"].notna().all(),
    "All risky events have incident IDs"
)

check(
    risk_labels["incident_id"].isin(
        incidents["incident_id"]
    ).all(),
    "All risky events reference known incidents"
)

incident_counts = (
    risk_labels.groupby("incident_id")
    .size()
)

check(
    len(incident_counts) == len(incidents)
    and incident_counts.eq(3).all(),
    "Every incident contains exactly 3 risky events"
)


# 7. Authorization consistency
actors_index = actors.set_index("actor_id")

authorization_matches = []

for event in events.itertuples(index=False):
    actor = actors_index.loc[event.actor_id]

    allowed_resources = json.loads(
        actor["allowed_resources"]
    )

    allowed_destinations = json.loads(
        actor["allowed_destinations"]
    )

    authorized = (
        event.resource_category in allowed_resources
        and event.destination_type in allowed_destinations
        and (
            event.actor_type != "ai_agent"
            or event.action == "tool_call"
        )
    )

    expected = (
        "authorized" if authorized else "unauthorized"
    )

    authorization_matches.append(
        event.authorization_status == expected
    )

check(
    all(authorization_matches),
    "Authorization states match actor permissions"
)


# 8. Analyze possible label shortcuts
merged = events.merge(
    labels[["event_id", "risk_label"]],
    on="event_id",
    validate="one_to_one"
)

print("\n=== AUTHORIZATION VS. RISK LABEL ===")

print(
    pd.crosstab(
        merged["authorization_status"],
        merged["risk_label"]
    )
)


# 9. Report
print("\n=== V3 VALIDATION SUMMARY ===")

if errors:
    print(f"FAILED: {len(errors)} checks")
    raise SystemExit(1)

print("PASS: All validation checks passed")
print("Events:", len(events))
print("Actors:", len(actors))
print("Incidents:", len(incidents))
