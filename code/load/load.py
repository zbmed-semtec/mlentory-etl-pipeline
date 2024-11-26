from kafka import KafkaConsumer
import json


def consume_messages(topic):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers="kafka:9092",
        api_version=(0, 11, 5),
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    for message in consumer:
        print(f"Received message: {message.value}")


# Example usage
consume_messages("example_topic")
