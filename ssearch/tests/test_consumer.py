import asyncio
import ray
from ray import serve
import ssearch.config as config

from aiokafka import AIOKafkaConsumer


@serve.deployment(num_replicas=2,
                  ray_actor_options={"num_cpus": 0.1, "num_gpus": 0},
                  health_check_period_s=1,
                  health_check_timeout_s=1)
class TestConsumer:
    def __init__(self, topic, model):
        self.model = model
        self.loop = asyncio.get_running_loop()

        self.consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=config.KAFKA_BROKER_URL,
            group_id=config.KAFKA_GROUP_ID,
            loop=self.loop,
            enable_auto_commit=True
        )
        self.healthy = True

        self.loop.create_task(self.consume())

    @serve.batch(max_batch_size=16)
    def __call__(self, batch: list[]):
        return self.model(*args, **kwargs)

    async def consume(self):
        await self.consumer.start()
        print("Consuming started")
        try:
            async for msg in self.consumer:
                print("Consuming message")
        finally:
            await self.consumer.stop()
            print("Consuming stopped")

    async def check_health(self):
        if not self.healthy:
            raise RuntimeError("Kafka Consumer is broken")
