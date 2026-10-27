
import sys
from pathlib import Path

import pandas as pd

# Allow imports from the project root
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.behavioral_features import build_behavioral_features


def test_historical_features():
    events = pd.DataFrame([
        {
            "event_id": "E1",
            "actor_id": "A001",
            "event_time": "2026-09-01T10:00:00Z",
            "size_mb": 10.0,
            "destination_type": "internal",
            "sensitivity": "internal",
        },
        {
            "event_id": "E2",
            "actor_id": "A001",
            "event_time": "2026-09-01T11:00:00Z",
            "size_mb": 20.0,
            "destination_type": "internal",
            "sensitivity": "internal",
        },
        {
            "event_id": "E3",
            "actor_id": "A001",
            "event_time": "2026-09-01T12:00:00Z",
            "size_mb": 30.0,
            "destination_type": "unapproved_external",
            "sensitivity": "confidential",
        },
    ])

    result = build_behavioral_features(events)

    assert result["actor_previous_events"].tolist() == [0, 1, 2]

    assert result["actor_previous_avg_size"].tolist() == [
        0.0, 10.0, 15.0
    ]

    assert result["actor_events_last_24h"].tolist() == [
        0, 1, 2
    ]

    assert result["is_unapproved_destination"].tolist() == [
        0, 0, 1
    ]

    assert result["is_high_sensitivity"].tolist() == [
        0, 0, 1
    ]


if __name__ == "__main__":
    test_historical_features()
    print("PASS: Historical feature tests")
