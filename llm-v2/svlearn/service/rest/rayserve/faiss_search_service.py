#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2023.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  #
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  #
#   Author: Asif Qamar
#  -------------------------------------------------------------------------------------------------
#
import logging as log

import ray
from ray import serve
from sentence_transformers import SentenceTransformer
from starlette.requests import Request

from svlearn.config import ConfigurationMixin
from svlearn.utils.compute_utils import get_port
from svlearn.text import FaissIndexer


@serve.deployment(
    # specify the number of GPU's available; zero if it is run on cpu
    ray_actor_options={"num_gpus": 0},
    # the number of instances of the  deployment in the cluster
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
    # the concurrency of the deployment
    max_concurrent_queries=1000,
)
class FaissSearchService(ConfigurationMixin):

    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.model_name = self.config['models']['multilingual-sentence-encoder']
        self.sentence_encoder = SentenceTransformer(self.model_name)

        index_file = self.config['search']['faiss']['index_file']
        if not index_file:
            raise ValueError(f'No index file specified in the config: {self.config}')

        self.faiss_indexer = FaissIndexer(index_file=index_file, dimension=512)

    async def __call__(self, request: Request):
        log.info(f"Received search request: {request}")
        payload = await request.json()
        queries: [str] = payload['queries']
        k: int = payload['k']
        if k is None:
            k = 10  # default to 10 results
        query_vectors = self.sentence_encoder.encode(queries)
        D, I = self.faiss_indexer.index.search(query_vectors, k=k)
        return {'results': I.tolist(), 'distances': D.tolist()}


if __name__ == '__main__':
    ray.init(address='ray://localhost:10001')

    mixin = ConfigurationMixin()
    config = mixin.load_config()
    url = config['services']['faiss_search']
    port = get_port(url)
    
    entrypoint = FaissSearchService.bind()  # bind() comes from the decorator
    serve.run(entrypoint, port=port, host="0.0.0.0", route_prefix="/faiss_search")
    log.info(f"Started serving FaissSearchService")
