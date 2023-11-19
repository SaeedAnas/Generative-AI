from qdrant_client import QdrantClient
import pandas as pd
import os

from config import config

client = QdrantClient("localhost", port=6333)

def extract_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

# Filename is in the format {name}.txt.parquet
# Want to extract {name}.txt as the filename
def read_parquet(path):
    """Reads a parquet file and returns a pandas dataframe and the filename"""
    filename = extract_filename(path)
    df = pd.read_parquet(path)

    return filename, df

    