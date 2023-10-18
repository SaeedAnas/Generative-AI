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
import logging as log
from typing import List, Tuple

from fastapi import FastAPI
import uvicorn

from svlearn.config.configuration import ConfigurationMixin
from svlearn.utils.compute_utils import get_port
from svlearn.text.query_results_reranker import QueryResultsReRanker
from svlearn.utils.reranker_class import RerankerInputRequest

class QueryResultsReRankerModel:
    """
    This class is the entry point for the query results reranker service.
    """
    def __init__(self):
        super().__init__()

    def initialize(self):
        self.encoder = QueryResultsReRanker()

    def __call__(self, request: RerankerInputRequest):
        """
        This method is called when the service is invoked,
         and it responds with reranked neighbours in order of descecnding score.
        :param request: the request object, with the query and potential neighbours
        :return: the reranked neighbours
        """
        log.info(f"Received request")
        query: str = request.query
        results: List[Tuple[int, str]] = request.results
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

app = FastAPI()

dispatcher = QueryResultsReRankerModel()

@app.post("/rerank")
async def rerank(request: RerankerInputRequest) :
    return dispatcher.__call__(request=request)

if __name__ == "__main__":
    import uvicorn
    dispatcher.initialize()

    mixin = ConfigurationMixin()
    config = mixin.load_config()
    url = config['services']['reranker']
    port = get_port(url) 
    
    uvicorn.run(app, host="localhost", port=port)
    log.info(f"Started serving results re-ranker service")

