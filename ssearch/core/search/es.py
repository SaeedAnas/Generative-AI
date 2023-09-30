import logging
import requests

from elasticsearch import Elasticsearch
from psycopg.rows import class_row

from ssearch.config import config
from ssearch.core.models import SearchResult

ELASTIC_URL = f"http://{config.ES_HOST}:{config.ES_PORT}"
ELASTIC_PASSWORD = config.ELASTIC_PASSWORD
ES_INDEX = config.ES_INDEX

es = Elasticsearch(
    ELASTIC_URL,
    basic_auth=("elastic", ELASTIC_PASSWORD)
)

def search_es(query, top_k):
    try:
        es_search = {
            "query": {
                "match": {
                    "chunk_text": query
                }
            },
            "size": top_k
        }
        
        es_results = es.search(index=ES_INDEX, body=es_search)
        results = [SearchResult.from_es_hit(hit) for hit in es_results["hits"]["hits"]]
        
        return results
    except Exception as e:
        logging.error(f"Error querying Elasticsearch: {e}")
        return []
    
if __name__ == "__main__":
    # Test the search function
    query = "What is diffusion?"
    top_k = 10
    es_results = search_es(query, top_k)
    print(es_results)
    # ids = search_faiss(query, top_k)
    # faiss_results = fetch_chunk_ids(ids)
    # print(faiss_results)
    # results = es_results + faiss_results
    # results = rerank(query, results)
    # print(results)