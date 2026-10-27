
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.temporal_features_v3_2 import add_temporal_features


def make_event(
    event_id,
    timestamp,
    actor_id,
    workflow_id,
    resource_id,
    size_mb,
):
    return {
        "event_id": event_id,
        "event_time": timestamp,
        "actor_id": actor_id,
        "workflow_id": workflow_id,
        "resource_id": resource_id,
        "size_mb": size_mb,
    }


def test_temporal_history():
    # Input intentionally shuffled to check sorting.
    events = pd.DataFrame([
        make_event(
            "E4", "2026-09-01T11:01:00Z",
            "A001", "W2", "DOC2", 40
        ),
        make_event(
            "E2", "2026-09-01T10:00:00Z",
            "A001", "W1", "DOC2", 20
        ),
        make_event(
            "E1", "2026-09-01T10:00:00Z",
            "A001", "W1", "DOC1", 10
        ),
        make_event(
            "E3", "2026-09-01T10:30:00Z",
            "A001", "W1", "DOC1", 30
        ),
        make_event(
            "E5", "2026-09-01T10:30:00Z",
            "A002", "W1", "DOC1", 100
        ),
    ])

    result = add_temporal_features(events)
    result = result.set_index("event_id")

    # Two same-time events must not see each other.
    assert result.loc["E1", "actor_events_last_1h"] == 0
    assert result.loc["E2", "actor_events_last_1h"] == 0

    assert (
        result.loc["E1", "actor_workflow_previous_events"]
        == 0
    )
    assert (
        result.loc["E2", "actor_workflow_previous_events"]
        == 0
    )

    # At 10:30, A001 has two prior events.
    assert result.loc["E3", "actor_events_last_1h"] == 2

    assert np.isclose(
        result.loc["E3", "actor_volume_last_1h_mb"],
        30.0,
    )

    assert (
        result.loc[
            "E3", "actor_unique_resources_last_1h"
        ] == 2
    )

    assert (
        result.loc["E3", "actor_resource_seen_before"]
        == 1
    )

    assert (
        result.loc[
            "E3", "actor_workflow_previous_events"
        ] == 2
    )

    assert np.isclose(
        result.loc[
            "E3", "actor_workflow_volume_before_mb"
        ],
        30.0,
    )

    # At 11:01, the 10:00 events are outside
    # the last-hour window.
    assert result.loc["E4", "actor_events_last_1h"] == 1

    assert np.isclose(
        result.loc["E4", "actor_volume_last_1h_mb"],
        30.0,
    )

    # The second actor must not inherit A001's history.
    assert result.loc["E5", "actor_events_last_1h"] == 0

    assert (
        result.loc[
            "E5", "actor_workflow_previous_events"
        ] == 0
    )

    assert (
        result.loc["E5", "actor_resource_seen_before"]
        == 0
    )

    print("PASS: Same-timestamp isolation")
    print("PASS: Rolling one-hour event counts")
    print("PASS: Rolling transfer volume")
    print("PASS: Historical resource tracking")
    print("PASS: Workflow history")
    print("PASS: Actor isolation")


def test_output_dataset():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data/v3_2"

    events = pd.read_csv(data_dir / "events.csv")
    features = pd.read_csv(
        data_dir / "temporal_features.csv"
    )

    assert len(events) == len(features)
    assert features["event_id"].is_unique

    assert set(events["event_id"]) == set(
        features["event_id"]
    )

    columns = [
        "actor_events_last_1h",
        "actor_volume_last_1h_mb",
        "actor_unique_resources_last_1h",
        "actor_workflow_previous_events",
        "actor_workflow_volume_before_mb",
        "actor_resource_seen_before",
    ]

    values = features[columns].to_numpy(
        dtype=float
    )

    assert np.isfinite(values).all()
    assert (values >= 0).all()

    print("PASS: V3.2 dataset integrity")


if __name__ == "__main__":
    test_temporal_history()
    test_output_dataset()

    print("\nPASS: All V3.2 temporal feature tests")
