
import json
import os
import urllib.error
import urllib.request

from confluent_kafka import Consumer, KafkaException


BROKER = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "127.0.0.1:9092",
)

TOPIC = "sentinel.activity.events"

API_URL = os.getenv(
    "SENTINEL_API_URL",
    "http://127.0.0.1:8000",
)

GROUP_ID = "sentinelai-scoring-v1"

# Process our initial five-message demonstration.
MAX_MESSAGES = 5


def submit_to_api(event):
    request = urllib.request.Request(
        url=f"{API_URL}/events/score",
        data=json.dumps(event).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=15,
        ) as response:
            return json.load(response)

    except urllib.error.HTTPError as error:
        body = error.read().decode(
            "utf-8", errors="replace"
        )

        # Our database uses unique event IDs.
        # For this local demo, an existing event
        # is treated as an already-processed event.
        if error.code == 409:
            print(
                "SKIPPED: Event already exists "
                f"in database: {event['event_id']}"
            )
            return {"duplicate": True}

        raise RuntimeError(
            f"FastAPI returned HTTP {error.code}: {body}"
        ) from error


def main():
    print("\n=== SENTINELAI KAFKA CONSUMER ===")
    print("Broker:", BROKER)
    print("Topic:", TOPIC)
    print("API:", API_URL)

    consumer = Consumer({
        "bootstrap.servers": BROKER,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })

    processed = 0
    idle_polls = 0

    try:
        consumer.subscribe([TOPIC])

        while processed < MAX_MESSAGES:
            message = consumer.poll(2.0)

            if message is None:
                idle_polls += 1

                if idle_polls >= 10:
                    print(
                        "No more messages received "
                        "during this demo."
                    )
                    break

                continue

            if message.error():
                raise KafkaException(message.error())

            idle_polls = 0

            event = json.loads(
                message.value().decode("utf-8")
            )

            print(
                f"\nRECEIVED: {event['event_id']} "
                f"offset={message.offset()}"
            )

            result = submit_to_api(event)

            if result.get("duplicate"):
                print("ALREADY PROCESSED")
            else:
                print(
                    "Risk score:",
                    round(result["risk_score"], 6)
                )
                print("Alert:", result["alert"])
                print(
                    "Prediction ID:",
                    result["prediction_id"]
                )
                print(
                    "Alert ID:",
                    result["alert_id"]
                )

            # Only commit after the API accepts
            # the event or confirms a duplicate.
            consumer.commit(
                message=message,
                asynchronous=False,
            )

            processed += 1

        print(
            f"\nPASS: Processed {processed} "
            "Kafka messages"
        )

    finally:
        consumer.close()


if __name__ == "__main__":
    main()
