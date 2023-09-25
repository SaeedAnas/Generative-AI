from pydantic import BaseSettings

import logging
# Write logs
logging.basicConfig(filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class Settings(BaseSettings):

    KAFKA_BROKER_URL: str
    KAFKA_TOPIC: str
    KAFKA_GROUP_ID: str
    FAISS_INDEX_PATH: str

    FILE_NAME_TMPL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
