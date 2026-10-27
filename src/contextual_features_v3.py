
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow imports when running from project root
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.behavioral_features import build_behavioral_features


DATA_DIR = Path("data/v3")


def build_contextual_features():
    # 1. Load raw events
    events = pd.read_csv(DATA_DIR / "events.csv")

    # 2. Reuse our tested historical feature logic
    df = build_behavioral_features(events)

    # 3. Add authorization feature
    df["is_unauthorized"] = (
        df["authorization_status"] == "unauthorized"
    ).astype(int)

    # 4. Add AI-agent indicator
    df["is_ai_agent"] = (
        df["actor_type"] == "ai_agent"
    ).astype(int)

    # 5. Add event-day feature
    df["event_day_of_week"] = (
        df["event_time"].dt.day_of_week
    )

    # 6. Create a simple sensitive-transfer indicator
    df["sensitive_external_transfer"] = (
        df["is_high_sensitivity"]
        * df["is_unapproved_destination"]
    )

    # 7. Convert infinite values to missing values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # 8. Validate important generated features
    required = [
        "event_id",
        "event_time",
        "actor_id",
        "actor_previous_events",
        "actor_previous_avg_size",
        "actor_size_ratio",
        "actor_events_last_24h",
        "is_cold_start",
        "is_unauthorized",
        "is_ai_agent",
    ]

    if df[required].isna().any().any():
        raise ValueError(
            "Missing values in required contextual features"
        )

    # 9. Save features separately from labels
    output = DATA_DIR / "contextual_features.csv"

    df.to_csv(output, index=False)

    print("\n=== V3 CONTEXTUAL FEATURES COMPLETE ===")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nFeature preview:")
    print(
        df[
            [
                "actor_id",
                "size_mb",
                "actor_previous_events",
                "actor_size_ratio",
                "actor_events_last_24h",
                "is_unauthorized",
                "is_ai_agent",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nSaved to:", output)


if __name__ == "__main__":
    build_contextual_features()
