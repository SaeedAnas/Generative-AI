import psycopg2
import os

DB_HOST = os.environ['DB_HOST']
DB_NAME = os.environ['DB_NAME']
DB_USER = os.environ['DB_USER']
DB_PASSWORD = os.environ['DB_PASSWORD']

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
cur = conn.cursor()

def create_tables():
    queries = [
        """
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            content TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE metadata (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            date_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );
        """,
        """
        CREATE TABLE chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_vector FLOAT8[] NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );
        """
    ]

    for query in queries:
        cur.execute(query)
    conn.commit()

if __name__ == "__main__":
    create_tables()
    print("Tables created successfully!")
    cur.close()
    conn.close()
