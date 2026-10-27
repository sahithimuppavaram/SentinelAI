
import pandas as pd

# Load our datasets
events = pd.read_csv("data/events.csv")
labels = pd.read_csv("data/labels.csv")

print("\n=== DATASET SHAPE ===")
print("Events:", events.shape)
print("Labels:", labels.shape)

print("\n=== COLUMN NAMES ===")
print(events.columns.tolist())

print("\n=== DATA TYPES ===")
print(events.dtypes)

print("\n=== MISSING VALUES ===")
print(events.isnull().sum())

print("\n=== DUPLICATE EVENT IDS ===")
print(events["event_id"].duplicated().sum())

print("\n=== ACTOR TYPES ===")
print(events["actor_type"].value_counts())

print("\n=== ACTION DISTRIBUTION ===")
print(events["action"].value_counts())

print("\n=== TRANSFER SIZE STATISTICS ===")
print(events["size_mb"].describe())

print("\n=== CLASS DISTRIBUTION ===")
print(labels["risk_label"].value_counts(normalize=True))

# Join labels only for offline analysis
analysis = events.merge(
    labels[["event_id", "risk_label"]],
    on="event_id",
    how="left",
    validate="one_to_one"
)

print("\n=== AVERAGE TRANSFER SIZE BY LABEL ===")
print(analysis.groupby("risk_label")["size_mb"].mean())

print("\n=== RISK LABEL BY DESTINATION ===")
print(pd.crosstab(
    analysis["destination_type"],
    analysis["risk_label"],
    normalize="index"
))
