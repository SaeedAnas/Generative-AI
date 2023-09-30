from pydantic import BaseSettings

import logging
# Write logs
logging.basicConfig(filename='app.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Config(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    ELASTIC_PASSWORD: str
    KIBANA_PASSWORD: str

    ES_INDEX: str
    ES_HOST: str
    ES_PORT: int
    KIBANA_PORT: int

    FAISS_PORT: int
    FAISS_HOST: str

    KAFKA_BROKER_URL: str
    KAFKA_TOPIC: str
    KAFKA_GROUP_ID: str
    FAISS_INDEX_PATH: str

    FILE_NAME_TMPL: str = "/tmp/my-partition-state-{tp.topic}-{tp.partition}.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = Config()
