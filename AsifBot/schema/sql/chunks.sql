CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT,
    chunk_vector FLOAT8 [] NOT NULL
);