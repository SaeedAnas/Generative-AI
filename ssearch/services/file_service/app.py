# Simple aiokafka watchdog producer
from aiokafka import AIOKafkaProducer
import asyncio
from concurrent.futures import ProcessPoolExecutor

import asyncio
from watchfiles import awatch, Change
from pydantic import BaseModel
import pathlib
import json
import logging
import glob

IMAGE_SUPPORTED = {
    ".jpg", ".jpeg", ".png"
}

AUDIO_SUPPORTED = {
    ".wav"
}

TEXT_TOPIC = "text-pipeline"
IMAGE_TOPIC = "image-pipeline"
AUDIO_TOPIC = "audio-pipeline"

FINISH_TOPIC = "file-finished"


class Subscription(BaseModel):
    path: str
    topic: str
    supported: set = None


class FileWatcher():
    def __init__(self, producer, loop):
        self.producer = producer
        self.loop = loop
        self.subscriptions = []

    @staticmethod
    async def create(host):
        producer = AIOKafkaProducer(bootstrap_servers=host)
        loop = asyncio.new_event_loop()

        return FileWatcher(
            producer=producer,
            loop=loop
        )

    def validate_ext(path, supported):
        if supported is None:
            return True

        ext = pathlib.Path(path).suffix
        return ext in supported

    def subscribe(self, subscription):
        self.subscriptions.append(subscription)

    async def start(self):
        await self.producer.start()

        executor = ProcessPoolExecutor(2)
        for subscription in self.subscriptions:
            self.loop.run_in_executor(
                executor, self.watch_directory(subscription))

        try:
            self.loop.run_forever()
        finally:
            await self.producer.stop()
            self.loop.close()

    @staticmethod
    def serialize(value):
        return json.dumps(value).encode()

    @staticmethod
    async def send(self, topic, path):
        data = {"path": path}
        await self.producer.send(
            topic,
            self.serialize(data)
        )

    async def watch_directory(self, sub):
        async for changes in awatch(sub.path):
            for change, path in changes:
                if change == Change.added:
                    if self.validate_ext(path, sub.supported):
                        logging.info(path)
                        await self.send(sub.topic, path)


async def main():
    watcher = await FileWatcher.create("localhost:9092")
    watcher.subscribe(Subscription(
        topic=TEXT_TOPIC,
        path="/Users/anassaeed/Documents/TEXT"
    ))
    watcher.subscribe(Subscription(
        topic=IMAGE_TOPIC,
        path="/Users/anassaeed/Documents/IMAGE",
        supported=IMAGE_SUPPORTED
    ))
    watcher.subscribe(Subscription(
        topic=AUDIO_TOPIC,
        path="/Users/anassaeed/Documents/AUDIO",
        supported=AUDIO_SUPPORTED
    ))

    await watcher.start()

if __name__ == "__main__":
    asyncio.run(main())
