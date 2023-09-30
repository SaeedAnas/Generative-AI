# Simple aiokafka watchdog producer
from aiokafka import AIOKafkaProducer
import asyncio
from concurrent.futures import ProcessPoolExecutor

import asyncio
from watchfiles import awatch, Change
from pydantic import BaseModel

IMAGE_SUPPORTED = {
    ".jpg", ".jpeg", ".png"
}

AUDIO_SUPPORTED = {
    ".wav"
}

TEXT_TOPIC = "text-pipeline"
IMG_TOPIC = "image-pipeline"
AUDIO_TOPIC = "audio-pipeline"

FINISH_TOPIC = "file-finished"


class Subscription(BaseModel):
    path: str
    topic: str
    supported: set


class FileManager():
    def __init__(self):
        self.producer = AIOKafkaProducer(bootstrap_servers="localhost:9092")
        self.subscriptions = []
        self.loop = asyncio.new_event_loop()

    async def start(self):
        await self.producer.start()
        
        exectuor = ProcessPoolExecutor


    async def watch_directory(self, path, topic, supported=None):
        async for changes in awatch(path):
            batch = self.producer.create_batch()
            for change, path in changes:
                if change == Change.added:


async def watch():
    async for changes in awatch("/Users/anassaeed/Documents/TEXT"):
        for change, path in changes:
            if change == Change.added:
                print(path)

asyncio.run(main())
