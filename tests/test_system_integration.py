
import json
import urllib.request

import psycopg

from src.investigation_agent import investigate
from src.database import get_connection


def test_api():
    with urllib.request.urlopen(
        "http://127.0.0.1:8000/health",
        timeout=10,
    ) as response:
        data = json.load(response)

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    print("PASS: FastAPI health check")


def test_database():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM events),
                    (SELECT COUNT(*) FROM predictions),
                    (SELECT COUNT(*) FROM alerts)
                """
            )
            events, predictions, alerts = cursor.fetchone()

    assert events >= 7
    assert predictions >= 7
    assert alerts >= 3

    print(
        f"PASS: PostgreSQL - {events} events, "
        f"{predictions} predictions, {alerts} alerts"
    )


def test_investigation():
    report = investigate(3)

    evidence = report["evidence"]

    assert report["actor_id"] == "E064"
    assert evidence["recorded_events"] == 2
    assert evidence["total_volume_mb"] == 145.69
    assert evidence["flagged_events"] == 2

    assert set(evidence["event_ids"]) == {
        "EV00013297",
        "EV00013298",
    }

    print("PASS: LangGraph investigation")
    print("PASS: Duplicate evidence excluded")


def main():
    print("\n=== SENTINELAI SYSTEM TEST ===\n")

    test_api()
    test_database()
    test_investigation()

    print("\nPASS: All integration checks completed")


if __name__ == "__main__":
    main()
