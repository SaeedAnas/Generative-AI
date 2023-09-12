# indexer.py

import faiss
import numpy as np
import psycopg2
from elasticsearch import Elasticsearch, helpers
import logging
import os

# Setup logging
logging.basicConfig(filename='indexer.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('indexer')

# Database connection parameters
DB_PARAMS = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'username'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

ELASTIC_URL = os.getenv("ELASTIC_URL")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
ELASTIC_CERT_PATH = os.getenv("ELASTIC_CERT_PATH")


# FAISS setup
DIMENSION = 768
faiss_index = faiss.IndexFlatL2(DIMENSION)

# Create the client instance
es = Elasticsearch(
    ELASTIC_URL,
    ca_certs=ELASTIC_CERT_PATH,
    basic_auth=("elastic", ELASTIC_PASSWORD)
)


def connect_to_db(params):
    """Establish a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**params)
        logger.info("Successfully connected to the database.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def index_data(conn):
    """Fetch data from the database and index it in Elasticsearch and FAISS."""
    logger.info("Starting indexing process...")

    cur = conn.cursor()
    
    try:
        # Fetch data from the database
        cur.execute("SELECT chunks.id, chunk_text, chunk_vector FROM chunks JOIN documents ON chunks.document_id = documents.id;")
        actions = []

        for row in cur.fetchall():
            chunk_id, chunk_text, chunk_vector = row
            chunk_vector_np = np.array(chunk_vector, dtype=np.float32)
            
            # Index in FAISS
            faiss_index.add(chunk_vector_np.reshape(1, DIMENSION))
            
            # Prepare data for Elasticsearch
            action = {
                "_index": "document_index",
                "_id": chunk_id,
                "_source": {
                    "text": chunk_text,
                    "vector": chunk_vector
                }
            }
            actions.append(action)

        # Bulk index in Elasticsearch
        helpers.bulk(es, actions)
        logger.info(f"Indexed {len(actions)} chunks into Elasticsearch.")

        # Save FAISS index to disk
        faiss.write_index(faiss_index, 'faiss_index.index')
        logger.info("Saved FAISS index to disk.")
    
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
    finally:
        cur.close()

if __name__ == "__main__":
    conn = connect_to_db(DB_PARAMS)
    if conn:
        index_data(conn)
        conn.close()
