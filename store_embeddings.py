import pandas as pd
import os
import glob

from asifbot import config
from asifbot.core.db import Postgres, Qdrant

pg = Postgres()
qd = Qdrant()

def extract_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

# Filename is in the format {name}.txt.parquet
# Want to extract {name}.txt as the filename
def read_parquet(path):
    """Reads a parquet file and returns a pandas dataframe and the filename"""
    filename = extract_filename(path)
    df = pd.read_parquet(path)
    
    return {
        "filename": filename,
        "df": df
    }

def read_embeddings_dir():
    embeddings_dir = os.path.join(config.DATA_DIR, "embeddings")
    # Get all the parquet files in the embeddings directory
    parquet_files = glob.glob(os.path.join(embeddings_dir, "*.parquet"))
    parquets = [read_parquet(path) for path in parquet_files]
    
    return parquets

def write_df_to_postgres(df):
    with pg.connect() as conn:
        data = [
            (row['chunk'], row['embedding'].tolist()) for _, row in df.iterrows()
        ]
        ids = []
        
        sql = "INSERT INTO chunks (chunk_text, chunk_vector) VALUES (%s, %s) RETURNING id;"
        
        for chunk, embedding in data:
            cur = conn.execute(sql, (chunk, embedding))
            id = cur.fetchone()[0]
            ids.append(id)
        
        conn.commit()
    
    return ids

def write_df_to_qdrant(df, ids):
    embeddings = df['embedding'].values.tolist()
    qd.upsert(embeddings, ids)


def store_embeddings(parquets):
    """First write to postgres, get the ids, then write to qdrant"""
    for parquet in parquets:
        filename = parquet['filename']
        df = parquet['df']
        
        chunk_ids = write_df_to_postgres(df)
        write_df_to_qdrant(df, chunk_ids)

if __name__ == "__main__":
    parquets = read_embeddings_dir()
    store_embeddings(parquets)
    