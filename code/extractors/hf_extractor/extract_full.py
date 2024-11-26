from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import json

# Configuration
BROKER = "kafka:9092"  # Kafka broker address
TOPIC_NAME = "hf_topic"


def send_message(topic, message):

    producer.send(topic, message)
    print("Sended message", message)

    # try:
    #     producer.send(topic, message)
    #     producer.flush()
    # except Exception as e:
    #         print(e)
    # finally:
    #     producer.close()


def check_and_create_topic(broker, topic_name, num_partitions=1, replication_factor=1):
    # Create an admin client
    admin_client = KafkaAdminClient(
        bootstrap_servers=broker, api_version=(0, 11, 5), client_id="admin"
    )

    # Check if the topic already exists
    existing_topics = admin_client.list_topics()
    print("Existing topics::::", existing_topics)
    if topic_name in existing_topics:
        print(f"Topic '{topic_name}' already exists.")
        return

    # Create the topic
    topic = NewTopic(
        name=topic_name,
        num_partitions=num_partitions,
        replication_factor=replication_factor,
    )

    try:
        admin_client.create_topics(new_topics=[topic], validate_only=False)
        print(f"Topic '{topic_name}' created successfully.")
    except Exception as e:
        print(f"Failed to create topic '{topic_name}': {e}")
    finally:
        admin_client.close()


# Call the function
# check_and_create_topic(BROKER, TOPIC_NAME)

# Example usage
# Creating topic
# check_and_create_topic(BROKER, TOPIC_NAME)
# Sending message through topic
producer = KafkaProducer(
    bootstrap_servers=BROKER,  # Update with Kafka broker address
    api_version=(0, 10, 2),
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)
send_message("hf_topic", {"key": "value11"})
send_message("hf_topic", {"key": "value22"})
send_message("hf_topic", {"key": "value33"})

producer.flush()
producer.close()
