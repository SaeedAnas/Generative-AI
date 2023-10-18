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

from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel
from typing import List

from svlearn.config.configuration import ConfigurationMixin
from svlearn.utils.compute_utils import get_port
from sentence_transformers import SentenceTransformer

from svlearn.text.faiss_indexer import FaissIndexer

class InputRequest(BaseModel):
    queries: List[str]
    k: int

class FaissSearchService(ConfigurationMixin):

    def __init__(self):
        super().__init__()

    def initialize(self): 
        self.config = self.load_config()
        self.model_name = self.config['models']['multilingual-sentence-encoder']
        self.sentence_encoder = SentenceTransformer(self.model_name)
        index_file = self.config['search']['faiss']['index_file']
        if not index_file:
            raise ValueError(f'No index file specified in the config: {self.config}')

        self.faiss_indexer = FaissIndexer(index_file=index_file, dimension=512)

    def __call__(self, request: InputRequest):
        """
        This method is called when the service is invoked,
         and it responds with nearest neighbours corresponding to the queries.
        :param request: the request object, with the query strings
        :return: the neighbors, as a list inside a json object
        """
        log.info(f"Received request")
        queries: [str] = request.queries
        k: int = request.k
        if k is None:
            k = 10  # default to 10 results
        query_vectors = self.sentence_encoder.encode(queries)
        D, I = self.faiss_indexer.index.search(query_vectors, k=k)
        return {'results': I.tolist(), 'distances': D.tolist()}


app = FastAPI()

dispatcher = FaissSearchService()

@app.post("/faiss_search")
async def faiss_search(request: InputRequest) :
    return dispatcher.__call__(request=request)

if __name__ == "__main__":
    import uvicorn
    dispatcher.initialize()

    mixin = ConfigurationMixin()
    config = mixin.load_config()
    url = config['services']['faiss_search']
    port = get_port(url) 
    
    uvicorn.run(app, host="localhost", port=port)
    log.info(f"Started serving Faiss-Search")

