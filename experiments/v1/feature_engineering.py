
from pathlib import Path

import numpy as np
import pandas as pd


def create_features(events: pd.DataFrame) -> pd.DataFrame:
    """Convert raw activity events into model features."""

    df = events.copy()

    # Convert event time to UTC datetime
    df["event_time"] = pd.to_datetime(
        df["event_time"],
        utc=True,
        errors="raise"
    )

    # Extract temporal features
    df["event_hour_utc"] = df["event_time"].dt.hour
    df["event_day_of_week"] = df["event_time"].dt.day_of_week

    # Transform numerical values
    df["log_size_mb"] = np.log1p(df["size_mb"])

    # Convert categories into numerical indicators
    df["is_high_sensitivity"] = df["sensitivity"].isin(
        ["confidential", "restricted"]
    ).astype(int)

    df["is_unapproved_destination"] = (
        df["destination_type"] == "unapproved_external"
    ).astype(int)

    df["is_upload"] = (
        df["action"] == "file_upload"
    ).astype(int)

    df["is_ai_agent"] = (
        df["actor_type"] == "ai_agent"
    ).astype(int)

    # For reproducible comparison, preserve IDs
    feature_columns = [
        "event_id",
        "event_time",
        "actor_id",
        "event_hour_utc",
        "event_day_of_week",
        "log_size_mb",
        "is_high_sensitivity",
        "is_unapproved_destination",
        "is_upload",
        "is_ai_agent",
    ]

    return df[feature_columns]


if __name__ == "__main__":
    events = pd.read_csv("data/events.csv")

    features = create_features(events)

    output_path = Path("data/features.csv")
    features.to_csv(output_path, index=False)

    print("\n=== FEATURE ENGINEERING COMPLETE ===")
    print("Rows:", len(features))
    print("Columns:", len(features.columns))
    print("\nFirst five rows:")
    print(features.head().to_string(index=False))
    print(f"\nSaved features to {output_path}")
