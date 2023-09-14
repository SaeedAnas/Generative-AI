# Searcher.py

import faiss
import numpy as np
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder
from helpers.log_utils import setup_logger
import os

# Set up the logger
logger = setup_logger()
logger.info("Begin of the Searcher")

ELASTIC_URL = os.getenv("ELASTIC_URL")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
ELASTIC_CERT_PATH = os.getenv("ELASTIC_CERT_PATH")

# Create the client instance
es = Elasticsearch(
    ELASTIC_URL,
    ca_certs=ELASTIC_CERT_PATH,
    basic_auth=("elastic", ELASTIC_PASSWORD)
)

# FAISS and Transformer setup
DIMENSION = 768
faiss_index = faiss.read_index('faiss_index.index')
print(f"Number of vectors in FAISS index: {faiss_index.ntotal}")

bi_encoder = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def search(query, top_k=10, rerank_k=5):
    """Execute the hybrid search mechanism with re-ranking."""
    logger.info(f"Searching for query: {query}")

    try:
        # Step 1: Use bi-encoder to get query vector
        query_vector = bi_encoder.encode(query, convert_to_numpy=True)

        # Step 2: Fetch top-k results from FAISS
        _, faiss_top_indices = faiss_index.search(query_vector.reshape(1, DIMENSION), top_k)
        
        # Step 3: Fetch detailed results from Elasticsearch using the indices
        es_results = es.mget(index="document_index", ids=faiss_top_indices[0].tolist())["docs"]

        # Step 4: Re-ranking top results using a cross-encoder
        pairs = [(query, hit["_source"]["text"]) for hit in es_results]
        scores = cross_encoder.predict(pairs)

        # Sort by scores
        sorted_results = sorted(zip(es_results, scores), key=lambda x: x[1], reverse=True)[:rerank_k]

        # Return top rerank_k results
        logger.info(f"Found {len(sorted_results)} relevant results for query: {query}")
        return [result[0]["_source"]["text"] for result in sorted_results]

    except Exception as e:
        logger.error(f"Error during search: {e}")
        return []

if __name__ == "__main__":
    query = input("Enter your search query: ").strip()
    search_results = search(query)
    
    # Printing the search results 
    for rank, (score, doc) in enumerate(search_results, 1):
        print(f"Rank: {rank}, Score: {score:.4f}")
        print(doc["_source"]["text"])
        print("-" * 80)
