# indexer.py

import faiss
import numpy as np
import psycopg2
from elasticsearch import Elasticsearch, helpers
from helpers.log_utils import setup_logger
import os

# Set up the logger
logger = setup_logger()
logger.info("Begin of the indexer")

import os

# Database connection parameters
DB_PARAMS = {
    'host': os.environ['DB_HOST'],
    'database': os.environ['DB_NAME'],
    'user': os.environ['DB_USER'],
    'password': os.environ['DB_PASSWORD']
}

ELASTIC_URL = os.environ['ELASTIC_URL']
ELASTIC_PASSWORD = os.environ['ELASTIC_PASSWORD']
ELASTIC_CERT_PATH = os.environ['ELASTIC_CERT_PATH']

# FAISS setup
DIMENSION = 768     #os.environ['DIMENSIONS']
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

    # Use IndexIDMap for the FAISS index
    base_index = faiss.IndexFlatL2(DIMENSION)
    index_id_map = faiss.IndexIDMap2(base_index)

    try:
        # Fetch data from the database
        cur.execute("SELECT chunks.id, chunk_text, chunk_vector FROM chunks JOIN documents ON chunks.document_id = documents.id;")
        actions = []

        vector_list = []  # To hold vectors
        id_list = []      # To hold corresponding IDs

        for row in cur.fetchall():
            chunk_id, chunk_text, chunk_vector = row
            chunk_vector_np = np.array(chunk_vector, dtype=np.float32)

            vector_list.append(chunk_vector_np)
            id_list.append(chunk_id)

            # Prepare data for Elasticsearch
            action = {
                "_index": "document_index",
                "_id": chunk_id,
                "_source": {
                    "text": chunk_text                    
                }
            }
            actions.append(action)

        # Convert lists to numpy arrays
        vectors_np = np.array(vector_list, dtype=np.float32)
        ids_np = np.array(id_list, dtype=np.int64)

        # Index in FAISS with IDs
        index_id_map.add_with_ids(vectors_np, ids_np)
            
        # Bulk index in Elasticsearch
        helpers.bulk(es, actions)
        logger.info(f"Indexed {len(actions)} chunks into Elasticsearch.")

        # Save FAISS index to disk
        faiss.write_index(index_id_map, 'faiss_index.index')
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
