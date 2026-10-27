
from pathlib import Path
import json
import os

import pandas as pd
from confluent_kafka import Producer


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/v3_2/splits/test.csv"

BROKER = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "127.0.0.1:9092",
)

TOPIC = "sentinel.activity.events"

# Start with just five events.
EVENT_LIMIT = 5


def delivery_report(error, message):
    """Called after Kafka acknowledges a message."""

    if error is not None:
        print(f"FAILED: {error}")
        delivery_errors.append(str(error))
    else:
        print(
            f"DELIVERED: "
            f"partition={message.partition()}, "
            f"offset={message.offset()}"
        )


delivery_errors = []


def main():
    print("\n=== SENTINELAI KAFKA PRODUCER ===")

    events = pd.read_csv(DATA_PATH)

    # Load the same feature list used by FastAPI.
    import joblib

    artifact = joblib.load(
        ROOT / "models/v3_2/xgboost.joblib"
    )

    feature_columns = artifact["feature_columns"]

    producer = Producer({
        "bootstrap.servers": BROKER,
        "client.id": "sentinelai-producer",
        "acks": "all",
    })

    # Select events with different model scores:
    # three routine/legitimate events and two
    # events that the model would flag.
    scores = artifact["model"].predict_proba(
        events[feature_columns]
    )[:, 1]

    threshold = float(artifact["threshold"])

    alert_positions = [
        i for i, score in enumerate(scores)
        if score >= threshold
    ][:2]

    normal_positions = [
        i for i, score in enumerate(scores)
        if score < threshold
    ][:3]

    selected = normal_positions + alert_positions

    assert len(selected) == EVENT_LIMIT

    for index in selected:
        row = events.iloc[index]

        payload = {
            "event_id": str(row["event_id"]),
            "actor_id": str(row["actor_id"]),
            "actor_type": str(row["actor_type"]),
            "event_time": str(row["event_time"]),
            "action": str(row["action"]),
            "resource_id": str(row["resource_id"]),
            "destination_type": str(
                row["destination_type"]
            ),
            "size_mb": float(row["size_mb"]),
            "features": {
                name: float(row[name])
                for name in feature_columns
            },
        }

        producer.produce(
            topic=TOPIC,
            key=payload["actor_id"].encode("utf-8"),
            value=json.dumps(payload).encode("utf-8"),
            callback=delivery_report,
        )

        print(
            f"QUEUED: event={payload['event_id']}, "
            f"actor={payload['actor_id']}"
        )

        # Process delivery callbacks.
        producer.poll(0)

    remaining = producer.flush(15)

    if remaining != 0 or delivery_errors:
        raise RuntimeError(
            "Not all Kafka messages were delivered"
        )

    print("\nPASS: Five events delivered to Kafka")
    print("Topic:", TOPIC)


if __name__ == "__main__":
    main()
