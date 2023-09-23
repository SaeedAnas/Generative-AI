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

# Constants and Global Variables
ELASTIC_URL = os.getenv("ELASTIC_URL")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
ELASTIC_CERT_PATH = os.getenv("ELASTIC_CERT_PATH")

MODEL_SBERT_768 = os.environ['MODEL_SBERT_768'] 
MODEL_SBERT_384 = os.environ['MODEL_SBERT_384'] 

# Create the client instance
es = Elasticsearch(
    ELASTIC_URL,
    ca_certs=ELASTIC_CERT_PATH,
    basic_auth=("elastic", ELASTIC_PASSWORD)
)

# FAISS and Transformer setup
DIMENSION = 768
try:
    faiss_index = faiss.read_index('faiss_index.index')
    logger.info(f"Number of vectors in FAISS index: {faiss_index.ntotal}")
except Exception as e:
    logger.error(f"Error reading FAISS index: {e}")
    exit(1)  # Exit if cannot read the index

bi_encoder = None
cross_encoder = None
try:
    bi_encoder = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')
    cross_encoder = CrossEncoder(MODEL_SBERT_768)
except Exception as e:
    logger.error(f"Error loading models: {e}")
    exit(1)  # Exit if cannot load the models

def search(query, top_k=10, rerank_k=5):
    """Execute the hybrid search with re-ranking."""
    if not query:
        logger.warning("Received an empty query. Aborting search.")
        return []

    logger.info(f"Searching for query: {query}")

    # Step 1: Lexical Search with Elasticsearch
    try:
        es_search = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": top_k  # get top_k results
        }

        es_results = es.search(index="document_index", body=es_search)
        candidate_sentences = [hit["_source"]["text"] for hit in es_results["hits"]["hits"]]

    except Exception as e:
        logger.error(f"Error during Elasticsearch search: {e}")
        return []

    # Step 2: Semantic Search with FAISS
    try:
        # Encode candidate sentences and query
        candidate_embeddings = bi_encoder.encode(candidate_sentences, convert_to_numpy=True)
        query_vector = bi_encoder.encode(query, convert_to_numpy=True)

        # Get closest vectors to the query
        _, faiss_indices = faiss_index.search(query_vector.reshape(1, DIMENSION), top_k)
        
        # Ensure we don't exceed the length of candidate_sentences
        max_index = len(candidate_sentences) - 1
        capped_indices = [i if i <= max_index else max_index for i in faiss_indices[0]]

        semantic_hits = [candidate_sentences[i] for i in capped_indices]

    except Exception as e:
        logger.error(f"Error during FAISS search: {e}")
        return []

    # Step 3: Re-ranking with Cross-encoder
    try:
        pairs = [(query, hit) for hit in semantic_hits]
        scores = cross_encoder.predict(pairs)

        # Sort by scores and select top rerank_k results
        sorted_results = sorted(zip(semantic_hits, scores), key=lambda x: x[1], reverse=True)[:rerank_k]

        logger.info(f"Found {len(sorted_results)} relevant results for query: {query}")
        return sorted_results

    except Exception as e:
        logger.error(f"Error during re-ranking: {e}")
        return []


if __name__ == "__main__":
    query = input("Enter your query: ").strip()
    if not query:
        logger.warning("Empty search query received.")
        exit(0)  # End the program if the query is empty

    search_results = search(query)

    # Printing the search results 
    for rank, (text, score) in enumerate(search_results, 1):
        print(f"Rank: {rank}, Score: {score:.4f}")
        print(text)
        print("-" * 80)


if __name__ == "__main__":
    query = input("what is spark: ").strip()
    if not query:
        logger.warning("Empty search query received.")
        exit(0)  # End the program if the query is empty

    search_results = search(query)
    
    # Printing the search results 
    for rank, (score, doc) in enumerate(search_results, 1):
#        print(f"Rank: {rank}, Score: {score:.4f}")
        print(f"Rank: {rank}, Score: {score if isinstance(score, str) else f'{score:.4f}'}")
#        print(doc["_source"]["text"])

        if isinstance(doc, dict) and "_source" in doc and "text" in doc["_source"]:
            print(doc["_source"]["text"])
        else:
            print("Unexpected format for 'doc'")

        print("-" * 80)