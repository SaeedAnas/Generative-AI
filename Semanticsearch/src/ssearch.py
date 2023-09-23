import logging
import requests

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
from psycopg.rows import class_row

from src.settings import settings
from src.helpers.db import db
from src.models import SearchResult

ELASTIC_URL = f"http://{settings.ES_HOST}:{settings.ES_PORT}"
ELASTIC_PASSWORD = settings.ELASTIC_PASSWORD
ES_INDEX = settings.ES_INDEX

es = Elasticsearch(
    ELASTIC_URL,
    basic_auth=("elastic", ELASTIC_PASSWORD)
)

FAISS_URL = f"http://{settings.FAISS_HOST}:{settings.FAISS_PORT}"

bi_encoder = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')


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
    
def search_faiss(query, top_k):
    query_vector = bi_encoder.encode(query, convert_to_numpy=True)

    faiss_search = {
        "vector": query_vector.tolist(),
        "top_k": top_k
    }
    
    try:
        faiss_results = requests.get(f"{FAISS_URL}/query", json=faiss_search)
        faiss_hits = faiss_results.json()["ids"]
        
        return faiss_hits
    except Exception as e:
        logging.error(f"Error querying FAISS: {e}")
        return []
    
def fetch_chunk_ids(ids):
    try:
        # conn is psycopg3 connection
        with db.connect() as conn:
            cur = conn.cursor(row_factory=class_row(SearchResult))
            cur.execute("SELECT id,document_id,chunk_text FROM chunks WHERE id = ANY(%s)", [ids])
            results = cur.fetchall()
            conn.commit()
            return results
    except Exception as e:
        logging.error(f"Error fetching chunk IDs: {e}")
        return []
    
def rerank(query, results: SearchResult):
    pairs = [(query, hit.chunk_text) for hit in results]
    print(pairs)
    scores = cross_encoder.predict(pairs)
    
    for result, score in zip(results, scores):
        result.score = score
    
    results.sort(key=lambda x: x.score, reverse=True)
    return results

def ssearch(query, top_k):
    es_results = search_es(query, top_k)
    ids = search_faiss(query, top_k)
    faiss_results = fetch_chunk_ids(ids)
    results = es_results + faiss_results
    results = rerank(query, results)
    return results
    
            
if __name__ == "__main__":
    # Test the search function
    query = "What is diffusion?"
    top_k = 10
    results = ssearch(query, top_k)
    print(results)
    # es_results = search_es(query, top_k)
    # print(es_results)
    # ids = search_faiss(query, top_k)
    # faiss_results = fetch_chunk_ids(ids)
    # print(faiss_results)
    # results = es_results + faiss_results
    # results = rerank(query, results)
    # print(results)