#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2023.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  #
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  #
#   Author: Chandar L
#  -------------------------------------------------------------------------------------------------
#
import asyncio
import logging as log
import signal
import sys
import time

import numpy as np
import uvicorn

from svlearn.config import ConfigurationMixin
from svlearn.text import FaissIndexer


import threading

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple

from svlearn.utils.compute_utils import get_port

class VectorRequest(BaseModel):
    vectors: List[Tuple[int, List[float]]]



class FaissIndexBuilderService(ConfigurationMixin):
    async def handle_shutdown(self):
        print("Received shutdown signal. Saving index before exiting...")
        await self.save_index()
        print("Index saved. Exiting...")
        sys.exit(0)

    def __init__(self):
        super().__init__()
        # Register a signal handler for SIGINT (Ctrl+C) and SIGTERM (kill)
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)        

    def initialize_index(self) :
        self.config = self.load_config()

        index_file = self.config['search']['faiss']['index_file']

        self.index_file = index_file
        self.faiss_indexer = FaissIndexer(index_file=index_file, dimension=512)
        self.counter = 0  # keep track of un-persisted updates to the index.
        self.start_time = time.time()
        self.elapsed_time = 0
        self.lock = threading.Lock()

    def __call__(self, request: VectorRequest):
        log.info(f"Received indexing request")
        vectors_with_ids: List[Tuple[int, List[float]]] = request.vectors
        if len(request.vectors) == 0:
            log.info(f"Received empty vectors list. Ignoring.")
            return {'status': 'nothing to index!'}

        log.info(f"Adding {len(vectors_with_ids)} vectors to the index")

        # extract the ids and vectors from the input
        ids = np.array([ind_id[0] for ind_id in vectors_with_ids])
        vectors_list: [np.array] = [np.array(ind_id[1]) for ind_id in vectors_with_ids]
        vectors = np.vstack(vectors_list).astype('float32')

        # train the faiss index if it is not trained already
        #
        # TODO: we need to think on how to periodically retrain the index,
        #  as the data is added to the index, and also how to pass it
        #  various parameters if needed, such as number of clusters for IVF, etc.
        #  For now, we will just train it once, and then add the vectors to it.
        try:
            # train the index if it is not trained already
            if not self.faiss_indexer.index.is_trained:
                self.faiss_indexer.index.train(vectors)
            # add the vectors to the index
            self.faiss_indexer.index.add_with_ids(vectors, ids)
        except Exception as e:
            log.error(f"Error while adding vectors to the faiss-index: {e}")
            raise e
        # persist the index to file every 10_000 updates or in 1 min interval
        self.counter += len(vectors_with_ids)
        self.elapsed_time = time.time() - self.start_time
        if self.counter > 10_000 or self.elapsed_time > 60:
            self.save_index()

        log.info(f"Added {len(vectors_with_ids)} vectors to the index")
        log.debug(f"Current index size: {self.faiss_indexer.size()}")
        return {'status': 'success'}

    def save_index(self):
        self.lock.acquire()
        log.info(f"Persisting the index to file: {self.index_file}")
        try:
            self.faiss_indexer.save_index(self.index_file)
            log.info(f"Index persisted to file: {self.index_file}")
        except Exception as e:
            log.error(f"Error while persisting the index to file: {self.index_file}. Error: {e}")
            raise e
        finally:
            self.counter = 0
            self.elapsed_time = 0
            self.start_time = time.time()
            self.lock.release()

app = FastAPI()
builder = FaissIndexBuilderService()

@app.post("/faiss_index_builder")
async def add_to_index(request: VectorRequest) :
    return builder.__call__(request=request)


if __name__ == "__main__":
    import uvicorn
    mixin = ConfigurationMixin()
    config = mixin.load_config()
    url = config['services']['faiss_index_builder']
    port = get_port(url) 

    builder.initialize_index()

    uvicorn.run(app, host="localhost", port=port)
    log.info(f"Started serving FaissIndexBuilderService")
