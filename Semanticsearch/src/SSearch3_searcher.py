# Searcher.py

import faiss
import numpy as np
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder
from helpers.log_utils import setup_logger
import os, sys

# Set up the logger
logger = setup_logger()
logger.info("Begin of the Searcher")

# Constants and Global Variables
ELASTIC_URL = os.getenv("ELASTIC_URL")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
ELASTIC_CERT_PATH = os.getenv("ELASTIC_CERT_PATH")

MODEL_SBERT = os.environ['MODEL_SBERT'] 

# Create the client instance
es = Elasticsearch(
    ELASTIC_URL,
    ca_certs=ELASTIC_CERT_PATH,
    basic_auth=("elastic", ELASTIC_PASSWORD)
)

# FAISS and Transformer setup
DIMENSION = 768 #os.environ['DIMENSIONS']
try:
    faiss_index = faiss.read_index('/Users/praveen/dev/project-SV/Assignment1/github/Generative-AI/Semanticsearch/src/faiss_index.index')
    logger.info(f"Number of vectors in FAISS index: {faiss_index.ntotal}")
except Exception as e:
    logger.error(f"Error reading FAISS index: {e}")
    exit(1)  # Exit if cannot read the index

bi_encoder = None
cross_encoder = None
try:
    bi_encoder = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
except Exception as e:
    logger.error(f"Error loading models: {e}")
    exit(1)  # Exit if cannot load the models


def search(query, top_k=10, rerank_k=5,search_type='text'):
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
        lexical_hits = [hit["_source"]["text"] for hit in es_results["hits"]["hits"]]

    except Exception as e:
        logger.error(f"Error during Elasticsearch search: {e}")
        return []

    # Step 2: Semantic Search with FAISS
    try:
        query_vector = bi_encoder.encode(query, convert_to_numpy=True)
        _, faiss_indices = faiss_index.search(query_vector.reshape(1, DIMENSION), top_k)
        semantic_hits = [es.get(index="document_index", id=index_id+1)["_source"]["text"] for index_id in faiss_indices[0]]

    except Exception as e:
        logger.error(f"Error during FAISS search: {e}")
        return []

    # Step 3: Combine Results
    combined_hits = list(set(semantic_hits[:top_k] + lexical_hits[:top_k]))

    # Step 4: Re-ranking with Cross-encoder
    try:
        pairs = [(query, hit) for hit in combined_hits]
        scores = cross_encoder.predict(pairs)
        normalized_scores = normalize_scores(scores)
        
        # Sort by scores and select top rerank_k results
        sorted_results = sorted(zip(combined_hits, normalized_scores), key=lambda x: x[1], reverse=True)[:rerank_k]
        #sorted_results = sorted(zip(combined_hits, scores), key=lambda x: x[1], reverse=True)[:rerank_k]
      
        logger.info(f"Found {len(sorted_results)} relevant results for query: {query}")

        # Check if we are in the context of an API request or standalone execution
        if 'fastapi' in sys.modules:
            # Return as a list of dictionaries for API compatibility
            results = [{"text": text, "score": float(score)} for text, score in sorted_results]
            #logger.info(f"Results: {results}")
            #print(f"Results: {results}")
            return results

        else:
            # Return as a list of tuples for standalone compatibility
            #return [(result["_source"]["text"], result["score"]) for result in sorted_results]
            return [(text, score) for text, score in sorted_results]
            logger.info(f"Results: {results}")
            #print(f"Results: {results}")
            #return results

    except Exception as e:
        logger.error(f"Error during re-ranking: {e}")
        return []

def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)

    # Avoiding division by zero if all scores are the same
    if min_score == max_score:
        return [0.5 for _ in scores]  # Or another default value, since all scores are the same
    
    return [(score - min_score) / (max_score - min_score) for score in scores]


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
