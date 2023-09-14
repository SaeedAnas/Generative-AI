import pytest
from src.helpers.text_utils import clean_text, chunk_text
from src.main import store_in_db, process_directory
import os
import psycopg2

# Sample data for testing
SAMPLE_TEXT = "Hello, world! This is a test."
CLEANED_SAMPLE_TEXT = "hello world this is a test"
SAMPLE_TEXT_CHUNKS = ["Hello, world!", "This is a test."]

# Setup a mock database connection for testing
DB_HOST = "localhost"  # Adjust as needed
DB_NAME = "test_db"    # Adjust as needed
DB_USER = "test_user"  # Adjust as needed
DB_PASSWORD = "test_password"  # Adjust as needed

# Sample file paths for testing
SAMPLE_FILE_PATH = os.path.join(os.path.dirname(__file__), "test_data/sample.txt")
LARGE_SAMPLE_FILE_PATH = os.path.join(os.path.dirname(__file__), "test_data/large_sample.txt")

@pytest.fixture(scope="module")
def setup_db():
    # Initialize a test DB connection
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()

    # Create tables if they don't exist (simplified for testing)
    cur.execute("CREATE TABLE IF NOT EXISTS documents (id SERIAL PRIMARY KEY, file_name VARCHAR, file_type VARCHAR, content TEXT);")
    cur.execute("CREATE TABLE IF NOT EXISTS metadata (id SERIAL PRIMARY KEY, document_id INTEGER REFERENCES documents(id), file_path TEXT);")
    cur.execute("CREATE TABLE IF NOT EXISTS chunks (id SERIAL PRIMARY KEY, document_id INTEGER REFERENCES documents(id), chunk_text TEXT, chunk_vector REAL[]);")

    yield conn, cur

    # Teardown (clean up test data)
    cur.execute("DROP TABLE chunks;")
    cur.execute("DROP TABLE metadata;")
    cur.execute("DROP TABLE documents;")
    cur.close()
    conn.close()

def test_clean_text():
    result = clean_text(SAMPLE_TEXT)
    assert result == CLEANED_SAMPLE_TEXT, f"Expected {CLEANED_SAMPLE_TEXT} but got {result}"

def test_chunk_text():
    chunks = chunk_text(SAMPLE_TEXT)
    assert chunks == SAMPLE_TEXT_CHUNKS, f"Expected chunks {SAMPLE_TEXT_CHUNKS} but got {chunks}"

def test_store_in_db(setup_db):
    conn, cur = setup_db
    store_in_db("sample", SAMPLE_TEXT, ".txt", SAMPLE_FILE_PATH)
    
    # Check if data was stored correctly
    cur.execute("SELECT content FROM documents WHERE file_name = 'sample';")
    result = cur.fetchone()
    assert result[0] == CLEANED_SAMPLE_TEXT, f"Expected {CLEANED_SAMPLE_TEXT} but got {result[0]}"

def test_process_directory(setup_db):
    conn, cur = setup_db
    process_directory(os.path.dirname(SAMPLE_FILE_PATH))
    
    # Check if the sample data was processed and stored correctly
    cur.execute("SELECT content FROM documents WHERE file_name = 'sample';")
    result = cur.fetchone()
    assert result[0] == CLEANED_SAMPLE_TEXT, f"Expected {CLEANED_SAMPLE_TEXT} but got {result[0]}"

# This test is for checking performance on larger files (optional and requires a sizable test file)
@pytest.mark.performance
def test_large_file_processing(setup_db):
    conn, cur = setup_db
    process_directory(os.path.dirname(LARGE_SAMPLE_FILE_PATH))
    
    # This is a basic test just to see if processing completes without errors for large files
    assert True


