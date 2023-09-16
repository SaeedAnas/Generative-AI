import os
import spacy
import numpy as np
import psycopg2
from tika import parser
from sentence_transformers import SentenceTransformer
from helpers.log_utils import setup_logger
from psycopg2 import pool
from tenacity import retry, wait_fixed, stop_after_attempt

# Constants and Global Variables
MODEL_SBERT_768 = os.environ['MODEL_SBERT_768'] 
MODEL_SBERT_384 = os.environ['MODEL_SBERT_384'] 
MODEL_SPACY = os.environ['MODEL_SPACY'] 

# Environment variables for DB connection
DB_HOST = os.environ['DB_HOST']
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']

nlp = spacy.load(MODEL_SPACY)
model = SentenceTransformer(MODEL_SBERT_384)

# Logger setup
logger = setup_logger()
logger.info("Initialization...")

# Connection pooling
db_pool = None

"""
Retry Logic: For getting a database connection, better to keep this here
If a connection fails, it will retry for 3 times, waiting 3 seconds between each attempt.
"""


@retry(wait=wait_fixed(3), stop=stop_after_attempt(3))
def get_connection():
    return db_pool.getconn()


def release_connection(conn):
    db_pool.putconn(conn)


def initialize_pool():
    global db_pool
    db_pool = psycopg2.pool.SimpleConnectionPool(
        1, 20, host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    if db_pool:
        logger.info("Database pool established.")
    else:
        logger.error("Failed to establish database pool.")
        raise Exception("Failed to establish database pool.")


def clean_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_punct]
    return ' '.join(tokens)


def chunk_text(text, threshold=0.4):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    sentence_embeddings = model.encode(
        sentences, batch_size=32, convert_to_numpy=True)

    chunks = []
    current_chunk = []
    for idx, sentence in enumerate(sentences[:-1]):
        current_chunk.append(sentence)
        
        current_embedding = sentence_embeddings[idx]
        next_embedding = sentence_embeddings[idx + 1]

        dist = np.linalg.norm(current_embedding - next_embedding)
        if dist > threshold:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def store_in_db(file_name, text, file_type, file_path, cursor):
    chunks = chunk_text(clean_text(text))
    cursor.execute("INSERT INTO documents (file_name, file_type, content) VALUES (%s, %s, %s) RETURNING id;",
                   (file_name, file_type, text))
    document_id = cursor.fetchone()[0]

    cursor.execute(
        "INSERT INTO metadata (document_id, file_path) VALUES (%s, %s) RETURNING id;", (document_id, file_path))
    for chunk in chunks:
        chunk_vector = model.encode(chunk).tolist()
        cursor.execute("INSERT INTO chunks (document_id, chunk_text, chunk_vector) VALUES (%s, %s, %s);",
                       (document_id, chunk, chunk_vector))


def process_directory(input_directory, conn):
    with conn.cursor() as cur:
        for root, _, files in os.walk(input_directory):
            for file in files:
                file_path = os.path.join(root, file)
                logger.info(f"Processing {file_path}")
                file_name, file_type = os.path.splitext(file)

                parsed = parser.from_file(file_path)
                content = parsed["content"]
                if content:
                    store_in_db(file_name, content, file_type, file_path, cur)
                conn.commit()  # we can commit at the end of each file for progress persistence


def main():
    try:
        initialize_pool()
        input_directory = input(
            "Enter the path to the directory containing the files: ").strip()
        conn = get_connection()
        process_directory(input_directory, conn)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except psycopg2.OperationalError:
        logger.error("Database operational error occurred. Retrying...")
    except Exception as e:
        logger.exception(f"Unexpected error during processing: {e}")
    finally:
        if db_pool:
            db_pool.closeall()
            logger.info("Closed all database connections.")


if __name__ == "__main__":
    main()
