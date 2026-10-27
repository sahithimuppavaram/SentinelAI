
from pathlib import Path

import numpy as np
import pandas as pd


def build_behavioral_features(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()

    # 1. Convert timestamps to UTC
    df["event_time"] = pd.to_datetime(
        df["event_time"],
        utc=True,
        errors="raise"
    )

    # 2. Sort events by actor and timestamp
    df = df.sort_values(
        ["actor_id", "event_time", "event_id"]
    ).reset_index(drop=True)

    # 3. Initialize historical feature columns
    df["actor_previous_events"] = 0
    df["actor_previous_avg_size"] = 0.0
    df["actor_events_last_24h"] = 0

    # 4. Calculate historical features for each actor
    for actor_id, group in df.groupby("actor_id"):

        # Convert timestamps into NumPy datetime64
        times = (
            group["event_time"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )

        # Transfer sizes for this actor
        sizes = group["size_mb"].to_numpy(dtype=float)

        # Find the first event with the current timestamp
        # This excludes ALL events at the same timestamp
        previous_end = np.searchsorted(
            times,
            times,
            side="left"
        )

        # Cumulative transfer volume
        cumulative_sizes = np.concatenate(
            ([0.0], np.cumsum(sizes))
        )

        # Number of strictly earlier events
        previous_counts = previous_end

        # Sum of sizes from strictly earlier events
        previous_sums = cumulative_sizes[previous_end]

        # Historical average (avoid division by zero)
        previous_averages = np.divide(
            previous_sums,
            previous_counts,
            out=np.zeros(len(group), dtype=float),
            where=previous_counts > 0
        )

        # Find start of trailing 24-hour window
        window_start = times - np.timedelta64(24, "h")

        window_start_indices = np.searchsorted(
            times,
            window_start,
            side="left"
        )

        # Count events from the past 24 hours,
        # excluding same-timestamp events
        previous_24h = (
            previous_end - window_start_indices
        )

        # Store historical features
        df.loc[
            group.index, "actor_previous_events"
        ] = previous_counts

        df.loc[
            group.index, "actor_previous_avg_size"
        ] = previous_averages

        df.loc[
            group.index, "actor_events_last_24h"
        ] = previous_24h

    # 5. Flag actors with fewer than 5 previous events
    df["is_cold_start"] = (
        df["actor_previous_events"] < 5
    ).astype(int)

    # 6. Compare current transfer with historical average
    # First event receives neutral ratio of 1.0
    df["actor_size_ratio"] = np.where(
        df["actor_previous_events"] > 0,
        df["size_mb"]
        / df["actor_previous_avg_size"].clip(lower=1.0),
        1.0
    )

    # 7. Extract event hour in UTC
    df["event_hour_utc"] = df["event_time"].dt.hour

    # 8. Identify unapproved external destinations
    df["is_unapproved_destination"] = (
        df["destination_type"] == "unapproved_external"
    ).astype(int)

    # 9. Identify confidential or restricted resources
    df["is_high_sensitivity"] = df["sensitivity"].isin(
        ["confidential", "restricted"]
    ).astype(int)

    # 10. Log-transform transfer size
    df["log_size_mb"] = np.log1p(df["size_mb"])

    return df


if __name__ == "__main__":

    # Load V2 dataset
    events = pd.read_csv("data/v2/events.csv")

    # Build behavioral features
    features = build_behavioral_features(events)

    # Create output directory if needed
    output = Path("data/v2/behavioral_features.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save engineered features
    features.to_csv(output, index=False)

    print("\n=== BEHAVIORAL FEATURE ENGINEERING COMPLETE ===")
    print("Rows:", len(features))
    print("Columns:", len(features.columns))

    print("\nSample historical features:")

    columns = [
        "actor_id",
        "event_time",
        "size_mb",
        "actor_previous_events",
        "actor_previous_avg_size",
        "actor_size_ratio",
        "actor_events_last_24h",
        "is_cold_start",
    ]

    print(
        features[columns]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nSaved to {output}")
