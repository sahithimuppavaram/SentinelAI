
import json
import os
import sys
from datetime import timedelta
from typing import TypedDict

import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END


class InvestigationState(TypedDict, total=False):
    alert_id: int
    alert: dict
    related_events: list[dict]
    report: dict


def get_database_connection():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is missing."
        )

    return psycopg.connect(
        database_url,
        row_factory=dict_row,
    )


def fetch_alert(state: InvestigationState):
    alert_id = state["alert_id"]

    with get_database_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    a.alert_id,
                    a.status,
                    a.created_at AS alert_created_at,
                    e.event_id,
                    e.actor_id,
                    e.actor_type,
                    e.event_time,
                    e.action,
                    e.resource_id,
                    e.destination_type,
                    e.size_mb,
                    p.model_name,
                    p.model_version,
                    p.risk_score,
                    p.threshold
                FROM alerts a
                JOIN predictions p
                    ON a.prediction_id = p.prediction_id
                JOIN events e
                    ON p.event_id = e.event_id
                WHERE a.alert_id = %s
                """,
                (alert_id,),
            )

            alert = cursor.fetchone()

    if alert is None:
        raise ValueError(
            f"Alert {alert_id} does not exist."
        )

    print(
        f"FETCHED ALERT: {alert_id} "
        f"for actor {alert['actor_id']}"
    )

    return {"alert": alert}


def collect_evidence(state: InvestigationState):
    alert = state["alert"]

    window_start = (
        alert["event_time"] - timedelta(hours=24)
    )

    with get_database_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                WITH ranked_events AS (
                    SELECT
                        e.event_id,
                        e.actor_id,
                        e.event_time,
                        e.action,
                        e.resource_id,
                        e.destination_type,
                        e.size_mb,
                        p.risk_score,
                        p.is_alert,
                        ROW_NUMBER() OVER (
                            PARTITION BY
                                e.actor_id,
                                e.actor_type,
                                e.event_time,
                                e.action,
                                e.resource_id,
                                e.destination_type,
                                e.size_mb
                            ORDER BY
                                CASE
                                    WHEN e.event_id = %s
                                    THEN 0
                                    WHEN e.event_id LIKE 'API-%%'
                                    THEN 2
                                    ELSE 1
                                END,
                                e.event_id
                        ) AS duplicate_rank
                    FROM events e
                    LEFT JOIN LATERAL (
                        SELECT
                            risk_score,
                            is_alert
                        FROM predictions
                        WHERE event_id = e.event_id
                        ORDER BY
                            predicted_at DESC,
                            prediction_id DESC
                        LIMIT 1
                    ) p ON TRUE
                    WHERE e.actor_id = %s
                      AND e.event_time >= %s
                      AND e.event_time <= %s
                )
                SELECT
                    event_id,
                    actor_id,
                    event_time,
                    action,
                    resource_id,
                    destination_type,
                    size_mb,
                    risk_score,
                    is_alert
                FROM ranked_events
                WHERE duplicate_rank = 1
                ORDER BY event_time, event_id
                """,
                (
                    alert["event_id"],
                    alert["actor_id"],
                    window_start,
                    alert["event_time"],
                ),
            )

            events = cursor.fetchall()

    print(
        f"COLLECTED EVIDENCE: {len(events)} "
        "distinct recorded activities"
    )

    return {"related_events": events}


def generate_report(state: InvestigationState):
    alert = state["alert"]
    events = state["related_events"]

    total_volume = sum(
        float(event["size_mb"])
        for event in events
    )

    high_risk_events = [
        event for event in events
        if event["is_alert"] is True
    ]

    unique_resources = {
        event["resource_id"]
        for event in events
        if event["resource_id"] is not None
    }

    actions = {}

    for event in events:
        action = event["action"]
        actions[action] = actions.get(action, 0) + 1

    report = {
        "alert_id": alert["alert_id"],
        "actor_id": alert["actor_id"],
        "actor_type": alert["actor_type"],
        "trigger_event_id": alert["event_id"],
        "alert_status": alert["status"],
        "model_name": alert["model_name"],
        "model_version": alert["model_version"],
        "risk_score": round(
            float(alert["risk_score"]), 6
        ),
        "threshold": round(
            float(alert["threshold"]), 6
        ),
        "investigation_window": {
            "hours": 24,
            "end_time": str(alert["event_time"]),
        },
        "evidence": {
            "recorded_events": len(events),
            "total_volume_mb": round(
                total_volume, 3
            ),
            "unique_resources": len(
                unique_resources
            ),
            "flagged_events": len(
                high_risk_events
            ),
            "action_counts": actions,
            "event_ids": [
                event["event_id"]
                for event in events
            ],
        },
        "evidence_limitations": [
            "Only activities currently stored in "
            "PostgreSQL are available.",
            "Matching synthetic test records are "
            "counted once in this report.",
            "No complete actor behavioral baseline "
            "has been established.",
            "No authorization or business-purpose "
            "verification has been performed.",
            "Identical activity fields do not "
            "necessarily prove two real-world "
            "events are duplicates.",
        ],
        "assessment": (
            "The model flagged the trigger event "
            "for analyst review. Recorded activity "
            "alone does not establish malicious "
            "intent, unauthorized access, or a "
            "deviation from a verified baseline."
        ),
        "recommended_action": (
            "Human analyst review required."
        ),
    }

    print("GENERATED INVESTIGATION REPORT")

    return {"report": report}


workflow = StateGraph(InvestigationState)

workflow.add_node("fetch_alert", fetch_alert)
workflow.add_node("collect_evidence", collect_evidence)
workflow.add_node("generate_report", generate_report)

workflow.add_edge(START, "fetch_alert")
workflow.add_edge("fetch_alert", "collect_evidence")
workflow.add_edge("collect_evidence", "generate_report")
workflow.add_edge("generate_report", END)

investigation_graph = workflow.compile()


def investigate(alert_id: int):
    result = investigation_graph.invoke({
        "alert_id": alert_id
    })

    return result["report"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python -m src.investigation_agent "
            "<alert_id>"
        )
        sys.exit(1)

    alert_id = int(sys.argv[1])

    print("\n=== SENTINELAI INVESTIGATION ===\n")

    report = investigate(alert_id)

    print("\n=== INVESTIGATION REPORT ===\n")
    print(json.dumps(report, indent=2, default=str))
