import logging
import requests

from sentence_transformers import SentenceTransformer, CrossEncoder
from psycopg.rows import class_row

from ssearch.config import config
from ssearch.core.db.postgres import db
from ssearch.core.models import SearchResult


FAISS_URL = f"http://{config.FAISS_HOST}:{config.FAISS_PORT}"

# bi_encoder = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')
bi_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


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
            cur.execute(
                "SELECT id,document_id,chunk_text FROM chunks WHERE id = ANY(%s)", [ids])
            results = cur.fetchall()
            conn.commit()
            return results
    except Exception as e:
        logging.error(f"Error fetching chunk IDs: {e}")
        return []


def search_vector(query, top_k=10):
    ids = search_faiss(query, top_k)
    faiss_results = fetch_chunk_ids(ids)
    return faiss_results


if __name__ == "__main__":
    # Test the search function
    query = "What is diffusion?"
    top_k = 10
    results = search_vector(query, top_k)
    print(results)
    # es_results = search_es(query, top_k)
    # print(es_results)
    # ids = search_faiss(query, top_k)
    # faiss_results = fetch_chunk_ids(ids)
    # print(faiss_results)
    # results = es_results + faiss_results
    # results = rerank(query, results)
    # print(results)
