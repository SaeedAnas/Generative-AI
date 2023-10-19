from kafka.admin import KafkaAdminClient, NewTopic

admin_client = KafkaAdminClient(
    bootstrap_servers="localhost:29092",
    client_id="admin"
)

topic_list = [
    "text-pipeline",
    "image-pipeline",
    "audio-pipeline",
]


def create_topics(topic_list):
    topics = [NewTopic(topic, num_partitions=1, replication_factor=1)
              for topic in topic_list]
    admin_client.create_topics(new_topics=topics, validate_only=False)


create_topics(topic_list)
