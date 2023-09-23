from psycopg_pool import ConnectionPool

conninfo = "dbname=postgres user=postgres password=example host=localhost port=5432"

SCHEMA = {
    "documents": """
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        file_name TEXT NOT NULL,
        file_type TEXT NOT NULL,
        content TEXT NOT NULL
    );
    """,
    "metadata": """
    CREATE TABLE metadata (
        id SERIAL PRIMARY KEY,
        document_id INTEGER NOT NULL,
        file_path TEXT NOT NULL,
        date_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (document_id) REFERENCES documents(id)
    );
    """,
    "chunks": """
    CREATE TABLE chunks (
        id SERIAL PRIMARY KEY,
        document_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_vector FLOAT8[] NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents(id)
    );
    
    ALTER TABLE chunks REPLICA IDENTITY FULL;
    """
}


class DB:
    def __init__(self, conninfo=conninfo, min_size=1, max_size=20, **kwargs):
        self.pool = ConnectionPool(
            conninfo=conninfo, min_size=min_size, max_size=max_size, **kwargs)

    def connect(self):
        return self.pool.connection()

    def close(self):
        self.pool.close()

    def create_tables(self):
        with self.connect() as conn:
            for table_schema in SCHEMA.values():
                conn.execute(table_schema)
            conn.commit()

    def drop_tables(self):
        with self.connect() as conn:
            for table_name in SCHEMA.keys():
                conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            conn.commit()


db = DB()


if __name__ == "__main__":
    db.drop_tables()
    db.create_tables()