
from pathlib import Path
from collections import Counter, deque
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.behavioral_features import build_behavioral_features


DATA_DIR = PROJECT_ROOT / "data/v3_2"
ONE_HOUR = pd.Timedelta(hours=1)


def add_temporal_features(df):
    """
    Calculate actor and workflow history using
    strictly earlier event timestamps.

    Events sharing a timestamp cannot see each other.
    """

    df = df.copy()

    df["event_time"] = pd.to_datetime(
        df["event_time"], utc=True
    )

    df = df.sort_values(
        ["actor_id", "event_time", "event_id"]
    ).reset_index(drop=True)

    new_columns = [
        "actor_events_last_1h",
        "actor_volume_last_1h_mb",
        "actor_unique_resources_last_1h",
        "actor_workflow_previous_events",
        "actor_workflow_volume_before_mb",
        "actor_resource_seen_before",
    ]

    for column in new_columns:
        df[column] = 0.0

    # ================================================
    # 1. Actor history
    # ================================================

    for actor_id, group in df.groupby(
        "actor_id", sort=False
    ):

        # Recent events stored in chronological order
        recent = deque()
        resource_counts = Counter()

        # All resources observed before current time
        previously_seen = set()

        # Process equal-time events together
        for timestamp, same_time in group.groupby(
            "event_time", sort=True
        ):

            cutoff = timestamp - ONE_HOUR

            # Remove events outside the last hour
            while recent and recent[0][0] < cutoff:
                old_time, old_size, old_resource = (
                    recent.popleft()
                )

                resource_counts[old_resource] -= 1

                if resource_counts[old_resource] == 0:
                    del resource_counts[old_resource]

            recent_count = len(recent)
            recent_volume = sum(
                item[1] for item in recent
            )
            unique_resources = len(resource_counts)

            indices = same_time.index

            df.loc[
                indices, "actor_events_last_1h"
            ] = recent_count

            df.loc[
                indices, "actor_volume_last_1h_mb"
            ] = recent_volume

            df.loc[
                indices, "actor_unique_resources_last_1h"
            ] = unique_resources

            df.loc[
                indices, "actor_resource_seen_before"
            ] = same_time["resource_id"].isin(
                previously_seen
            ).astype(int).to_numpy()

            # Only now add current timestamp's events
            for record in same_time.itertuples():
                size = float(record.size_mb)
                resource = record.resource_id

                recent.append(
                    (timestamp, size, resource)
                )

                resource_counts[resource] += 1
                previously_seen.add(resource)

    # ================================================
    # 2. Workflow history
    # ================================================

    for _, group in df.groupby(
        ["actor_id", "workflow_id"],
        sort=False
    ):

        previous_count = 0
        previous_volume = 0.0

        for _, same_time in group.groupby(
            "event_time", sort=True
        ):

            indices = same_time.index

            df.loc[
                indices,
                "actor_workflow_previous_events"
            ] = previous_count

            df.loc[
                indices,
                "actor_workflow_volume_before_mb"
            ] = previous_volume

            # Update after processing all same-time events
            previous_count += len(same_time)
            previous_volume += float(
                same_time["size_mb"].sum()
            )

    return df


def main():
    print("\n=== BUILDING V3.2 TEMPORAL FEATURES ===")

    events = pd.read_csv(
        DATA_DIR / "events.csv"
    )

    # Reuse historical features from V2/V3
    features = build_behavioral_features(events)

    # Add more detailed temporal features
    features = add_temporal_features(features)

    # Authorization indicator
    features["is_unauthorized"] = (
        features["authorization_status"]
        == "unauthorized"
    ).astype(int)

    # AI-agent indicator
    features["is_ai_agent"] = (
        features["actor_type"] == "ai_agent"
    ).astype(int)

    features["event_day_of_week"] = (
        features["event_time"].dt.day_of_week
    )

    features["sensitive_external_transfer"] = (
        features["is_high_sensitivity"]
        * features["is_unapproved_destination"]
    )

    # ================================================
    # 3. Validate outputs
    # ================================================

    assert len(features) == len(events)

    assert features["event_id"].is_unique

    assert set(features["event_id"]) == set(
        events["event_id"]
    )

    temporal_columns = [
        "actor_events_last_1h",
        "actor_volume_last_1h_mb",
        "actor_unique_resources_last_1h",
        "actor_workflow_previous_events",
        "actor_workflow_volume_before_mb",
        "actor_resource_seen_before",
    ]

    values = features[
        temporal_columns
    ].to_numpy(dtype=float)

    assert np.isfinite(values).all()
    assert (values >= 0).all()

    output = (
        DATA_DIR / "temporal_features.csv"
    )

    features.to_csv(output, index=False)

    print("Total events:", len(features))
    print("Feature columns:", len(features.columns))

    print("\nNew temporal feature preview:")
    print(
        features[
            ["actor_id", "event_time"]
            + temporal_columns
        ].head(10).to_string(index=False)
    )

    print("\nPASS: Temporal features generated")
    print("Saved to:", output)


if __name__ == "__main__":
    main()
