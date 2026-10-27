
import pandas as pd

# 1. Load V2 datasets
events = pd.read_csv("data/v2/events.csv")
labels = pd.read_csv("data/v2/labels.csv")

# 2. Validate the dataset
assert len(events) == 10000
assert events["event_id"].is_unique
assert labels["event_id"].is_unique
assert set(events["event_id"]) == set(labels["event_id"])
assert events["size_mb"].ge(0).all()

# 3. Join events with labels for analysis only
df = events.merge(
    labels[["event_id", "risk_label", "scenario_type"]],
    on="event_id",
    validate="one_to_one",
)

df["event_time"] = pd.to_datetime(
    df["event_time"], utc=True
)

# 4. Check scenarios
print("\n=== SCENARIO DISTRIBUTION ===")
print(df["scenario_type"].value_counts())

# 5. Compare transfer sizes by scenario
print("\n=== TRANSFER SIZE BY SCENARIO ===")
print(
    df.groupby("scenario_type")["size_mb"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

# 6. Compare destination patterns
print("\n=== DESTINATION BY SCENARIO (%) ===")
print(
    pd.crosstab(
        df["scenario_type"],
        df["destination_type"],
        normalize="index"
    ).mul(100).round(2)
)

# 7. Compare large-transfer rates
df["large_transfer"] = df["size_mb"] >= 100

print("\n=== LARGE TRANSFERS (100+ MB) ===")
print(
    pd.crosstab(
        df["scenario_type"],
        df["large_transfer"],
        normalize="index"
    ).mul(100).round(2)
)

# 8. Look for legitimate large transfers
print("\n=== EXAMPLES: LEGITIMATE LARGE TRANSFERS ===")

legitimate_large = df[
    (df["scenario_type"] == "legitimate_unusual")
    & (df["size_mb"] >= 100)
]

print(
    legitimate_large[
        ["event_id", "actor_id", "size_mb",
         "destination_type", "risk_label"]
    ].head(5).to_string(index=False)
)

# 9. Look for risk-worthy smaller transfers
print("\n=== EXAMPLES: RISK-WORTHY SMALLER TRANSFERS ===")

risky_small = df[
    (df["risk_label"] == 1)
    & (df["size_mb"] < 100)
]

print(
    risky_small[
        ["event_id", "actor_id", "size_mb",
         "destination_type", "risk_label"]
    ].head(5).to_string(index=False)
)

print("\n=== V2 ANALYSIS COMPLETE ===")
