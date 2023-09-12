import os
import re
from tika import parser
from sentence_transformers import SentenceTransformer
import psycopg2
from log_utils import setup_logger
import numpy as np

# Set up the logger
logger = setup_logger()
logger.info("Begin of the store.")

# Initialize SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')

# Extracting environment variables
DB_HOST = os.environ['DB_HOST']
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()
except Exception as e:
    logger.error(f"Error connecting to the database: {e}")
    exit()

# Text Preprocessing
def clean_text(text):
    # Remove any HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
"""
# Splitting Text into Chunks
def chunk_text(text):
    sentences = text.split('.')
    return sentences
"""
def chunk_text(text, threshold=0.4):
    # Split the document into individual sentences
    sentences = re.split(r'(?<=[.!?])\s+', text) 
    sentence_embeddings = model.encode(sentences, batch_size=32, convert_to_numpy=True)  # Batch for efficiency
    
    chunks = []
    current_chunk = []
    for idx in range(len(sentences) - 1):
        current_chunk.append(sentences[idx])
        
        current_embedding = sentence_embeddings[idx]
        next_embedding = sentence_embeddings[idx + 1]
        
        dist = np.linalg.norm(current_embedding - next_embedding)
        if dist > threshold:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    # Add any remaining sentences to chunks
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Store Chunks in PostgreSQL
def store_in_db(file_name, text, file_type, file_path):
    """ Store parsed document details in PostgreSQL database """
    try:
        cur.execute("INSERT INTO documents (file_name, file_type, content) VALUES (%s, %s, %s) RETURNING id;", (file_name, file_type, text))
        document_id = cur.fetchone()[0]

        # Store in metadata table
        cur.execute("INSERT INTO metadata (document_id, file_path) VALUES (%s, %s) RETURNING id;", (document_id, file_path))
        metadata_id = cur.fetchone()[0]  # Even though we don't use this, we still retrieve it to ensure no error

        # Store in chunks table
        chunks = chunk_text(text)
        for chunk in chunks:
            chunk_vector = model.encode(chunk).tolist()
            cur.execute("INSERT INTO chunks (document_id, chunk_text, chunk_vector) VALUES (%s, %s, %s);", (document_id, chunk, chunk_vector))

        conn.commit()
    except Exception as e:
        logger.exception(f"Error storing data in the database for {file_name}: {e}")

# Parse Documents and Store in DB
def process_directory(input_directory):
    """ Process each document in the specified directory and store its details in the database """
    for root, _, files in os.walk(input_directory):
        for file in files:
            file_path = os.path.join(root, file)
            logger.info(f"Processing {file_path}")

            try:
                file_name, file_type = os.path.splitext(file)  # This will separate the file's name and its extension
            
                parsed = parser.from_file(file_path)
                content = parsed["content"]
                    
                if content:
                    cleaned_content = clean_text(content)
                    store_in_db(file_name, cleaned_content, file_type, file_path)
            except Exception as e:
                logger.exception(f"Error processing file {file_path}: {e}")


if __name__ == "__main__":
    input_directory = input("Enter the path to the directory containing the files: ").strip()
    
    process_directory(input_directory)
    
    # Closing database connections
    try:
        cur.close()
        conn.close()
    except Exception as e:
        logger.exception(f"Error closing the database connection: {e}")
