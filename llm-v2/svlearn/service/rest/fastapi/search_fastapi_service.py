# -------------------------------------------------------------------------------------------------
#  Copyright (c) 2023.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar
# -------------------------------------------------------------------------------------------------

import json
from typing import List, Tuple
import requests
import logging as _log
from fastapi import FastAPI
from svlearn.common.svexception import SVError

from svlearn.config import ConfigurationMixin
from svlearn.utils.compute_utils import _get_connection, _get_elastic_client, get_port

app = FastAPI()


class HybridSearch(ConfigurationMixin):
    def __init__(self):
        super().__init__()

    def initialize(self):
        self.config = self.load_config()
        self.ann_index = self.config['services']['faiss_search']
        self.es_client, self.es_index_name = _get_elastic_client()
        self.reranker = self.config['services']['reranker']
        self.connection = _get_connection()

    def faiss_search(self, query: str, k: int = 20):
        request_json = {"queries": [query], "k": k}
        response = requests.post(url=self.ann_index, json=request_json)
        response_json = response.json()
        return response_json['results']

    def elastic_search(self, query: str, k: int = 20):
        search_body = ({
                "size": k,
                "query": {
                    "match": {
                        "text": query
                    }
                }
            })
        es_search_results = self.es_client.search(index=self.es_index_name, body=search_body)
        es_search_result_ids = [hit["_id"] for hit in es_search_results['hits']['hits']]
        return es_search_result_ids
    
    def get_associated_chunks(self, result_ids: [int]):
        try:
            cursor = self.connection.cursor()
            query = "SELECT TEXT FROM CHUNK WHERE ID IN %s"
            id_tuple = tuple(result_ids)
            cursor.execute(query, (id_tuple,))
            column_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            text_id = column_names.index('TEXT')
            chunks = [row[text_id] for row in rows]
            return chunks
        except Exception as e:
            _log.error(f'Error while returning chunks: {e}')
            raise SVError(f'Error while returning chunks: {e}')    
        finally:
            cursor.close()
    
    def call_reranker(self, query: str, results: List[Tuple[int, str]]):

        reranker_input_json = {
            "query": query, 
            "results": results
            }
        item_json = json.dumps(reranker_input_json)
        headers = {"Content-Type": "application/json"}
        response = requests.post(url=self.reranker, data=item_json, headers=headers)        

        response_json = response.json()
        return response_json['re_ranked_results']
    
    def hybrid_search(self, query: str, k: int = 10):
        all_results = []
        es_results = [int(x) for x in self.elastic_search(query=query, k=2*k)]
        faiss_results = [int(x) for x in self.faiss_search(query=query, k=2*k)[0]]
        all_results.extend(es_results)
        all_results.extend(faiss_results)
        all_results = list(set(all_results))

        chunks = self.get_associated_chunks(result_ids=all_results)
        reranker_input = ([[x[0], x[1]]
                          for x in zip(all_results, chunks)])
        reranked_chunks = self.call_reranker(query=query, 
                                             results=reranker_input)

        return {'neighbours': reranked_chunks}

dispatcher = HybridSearch()

@app.get("/search")
async def search(query: str, k: int = 10):
    return dispatcher.hybrid_search(query=query, k=k)

if __name__ == "__main__":
    import uvicorn
    dispatcher.initialize()
    
    mixin = ConfigurationMixin()
    config = mixin.load_config()
    url = config['services']['search']
    port = get_port(url)  

    uvicorn.run(app, host="localhost", port=port)
    _log.info(f"Started serving HybridSearch")
    