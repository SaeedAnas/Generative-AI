import asyncio
from aiokafka import AIOKafkaConsumer, ConsumerRebalanceListener
from aiokafka.errors import OffsetOutOfRangeError

import json
import pathlib
from collections import Counter

from settings import settings
import logging

import numpy as np

FILE_NAME_TMPL = settings.FILE_NAME_TMPL

class RebalanceListener(ConsumerRebalanceListener):

    def __init__(self, consumer, local_state):
        self.consumer = consumer
        self.local_state = local_state

    async def on_partitions_revoked(self, revoked):
        print("Revoked", revoked)
        self.local_state.dump_local_state()

    async def on_partitions_assigned(self, assigned):
        print("Assigned", assigned)
        self.local_state.load_local_state(assigned)
        for tp in assigned:
            last_offset = self.local_state.get_last_offset(tp)
            if last_offset < 0:
                await self.consumer.seek_to_beginning(tp)
            else:
                self.consumer.seek(tp, last_offset + 1)


class LocalState:

    def __init__(self):
        self._counts = {}
        self._offsets = {}

    def dump_local_state(self):
        for tp in self._counts:
            fpath = pathlib.Path(FILE_NAME_TMPL.format(tp=tp))
            with fpath.open("w+") as f:
                json.dump({
                    "last_offset": self._offsets[tp],
                    "counts": dict(self._counts[tp])
                }, f)

    def load_local_state(self, partitions):
        self._counts.clear()
        self._offsets.clear()
        for tp in partitions:
            fpath = pathlib.Path(FILE_NAME_TMPL.format(tp=tp))
            state = {
                "last_offset": -1,  # Non existing, will reset
                "counts": {}
            }
            if fpath.exists():
                with fpath.open("r+") as f:
                    try:
                        state = json.load(f)
                    except json.JSONDecodeError:
                        pass
            self._counts[tp] = Counter(state['counts'])
            self._offsets[tp] = state['last_offset']

    def add_counts(self, tp, counts, last_offset):
        self._counts[tp] += counts
        self._offsets[tp] = last_offset

    def get_last_offset(self, tp):
        return self._offsets[tp]

    def discard_state(self, tps):
        for tp in tps:
            self._offsets[tp] = -1
            self._counts[tp] = Counter()


async def save_state_every_second(local_state):
    while True:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            break
        local_state.dump_local_state()

class KafkaConsumer:
    def __init__(self, bootstrap_servers, group_id, topic, faiss_handler, loop):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topic = topic
        self.faiss_handler = faiss_handler

        self.consumer = AIOKafkaConsumer(
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=False,
            auto_offset_reset="none",
            key_deserializer=lambda key: key.decode("utf-8") if key else "",
            value_deserializer=json.loads,
            loop=loop
        )
        
        self.local_state = LocalState()
        self.listener = RebalanceListener(self.consumer, self.local_state)
        logging.info("Kafka consumer initialized")

    async def consume(self):
        await self.consumer.start()

        self.consumer.subscribe(topics=[self.topic], listener=self.listener)
        
        save_task = asyncio.create_task(save_state_every_second(self.local_state))

        try:
            while True:
                try:
                    msg_set = await self.consumer.getmany(timeout_ms=1000)

                except OffsetOutOfRangeError as err:
                    # This means that saved file is outdated and should be
                    # discarded
                    tps = err.args[0].keys()
                    self.local_state.discard_state(tps)
                    await self.consumer.seek_to_beginning(*tps)
                    continue
                
                for tp, msgs in msg_set.items():
                    vectors = []
                    ids = []
            
                    counts = Counter()
                    for msg in msgs:
                        value = msg.value["payload"]["after"]
                        vectors.append(value["chunk_vector"])
                        ids.append(value['id'])
                        counts[msg.key] += 1
                        
                    vectors = np.array(vectors)
                    ids = np.array(ids)
                    
                    await self.faiss_handler.add_with_ids(vectors, ids)
                    self.local_state.add_counts(tp, counts, msg.offset)
                    logging.info(f"Processed {len(msgs)} messages from {tp}")
        except Exception as e:
            logging.error(f"Error during consuming: {e}")
                    
        finally:
            save_task.cancel()
            await save_task
            # await self.consumer.stop()
