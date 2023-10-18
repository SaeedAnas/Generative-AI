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
from typing import List, Tuple

import ray
from ray import serve
from starlette.requests import Request

from svlearn.config.configuration import ConfigurationMixin
from svlearn.utils.compute_utils import get_port
from svlearn.text.query_results_reranker import QueryResultsReRanker


@serve.deployment(
    # specify the number of GPU's available; zero if it is run on cpu
    ray_actor_options={"num_gpus": 0},
    # the number of instances of the  deployment in the cluster
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
    # the concurrency of the deployment
    max_concurrent_queries=100,
)
class QueryResultsReRankerModel:
    """
    This class is the entry point for the query results reranker service.
    """
    def __init__(self):
        super().__init__()
        self.encoder = QueryResultsReRanker()

    async def __call__(self, request: Request):
        """
        This method is called when the service is invoked,
         and it responds with embedding vectors corresponding to the sentences.
        :param request: the request object, with the sentences to be encoded
        :return: the embedding vectors, as a list inside a json object
        """
        log.info(f"Received request: {request}")
        payload = await request.json()
        query: str = payload['query']
        results: List[Tuple[int,str]] = payload['results']
        try:
            log.info(f"Re-ranking  {len(results)} results against the query: {query}")
            log.debug(f"Results to re-rank: {results}")

            re_ranked_results = self.encoder.rerank(query, results)

            log.info(f"Returning re-ranked results: {len(re_ranked_results)} for query: {query}")
            log.debug(f"Re-ranked results: {re_ranked_results} for query: {query}")

            return {'re_ranked_results': re_ranked_results}
        except Exception as e:
            log.error(f'Error while reranking the search results: {e}. '
                      f'The query: {query}, the results were: {results}')
            raise e


if __name__ == '__main__':
    ray.init(address='ray://localhost:10001')

    mixin = ConfigurationMixin()
    config = mixin.load_config()
    url = config['services']['reranker']
    port = get_port(url)

    entrypoint = QueryResultsReRankerModel.bind()  # bind() comes from the decorator
    serve.run(entrypoint, port=port, host="0.0.0.0", route_prefix="/rerank")
    log.info(f"Started serving QueryResultsReRankerModel service")
