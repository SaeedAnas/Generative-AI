import psycopg
from helpers.db import db
import os
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import spacy
import glob
from tika import parser
import pandas as pd
from pyspark.sql.types import StringType, ArrayType, FloatType, IntegerType, Row
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

import findspark
findspark.init()

conninfo = "dbname=postgres user=postgres password=example host=localhost port=5432"


def get_files(path):
    """
    Get all files in a directory.
    """
    files = glob.glob(f"{path}/**/*", recursive=True)
    return [f for f in files if os.path.isfile(f)]


def parse_file(file):
    """
    Get duration of media file using MediaInfo.
    """
    parsed = parser.from_file(file)
    return parsed


def checkpoint(start):
    """
    Checkpoint time.
    """
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")


def remove_punctuation_and_stopwords(text):
    tokens = [
        token.text for token in text if not token.is_punct and not token.is_stop]
    return ' '.join(tokens)


def remove_punctuation(text):
    tokens = [token.text for token in text if not token.is_punct]
    return ' '.join(tokens)


def extract_sentences(text):
    # sentences = [remove_punctuation_and_stopwords(sent) for sent in text.sents]
    sentences = [remove_punctuation_and_stopwords(sent) for sent in text.sents]
    vectors = np.stack([sent.vector / sent.vector_norm for sent in text.sents])

    return sentences, vectors


def chunk_text_spacy(sentences, vectors, threshold=0.7):
    chunks = []
    current_chunk = []

    for idx in range(len(sentences) - 1):
        current_chunk.append(sentences[idx])

        current_embedding = vectors[idx]
        next_embedding = vectors[idx + 1]

        dist = np.linalg.norm(current_embedding - next_embedding)
        if dist > threshold:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    return chunks


def write_to_db(row, conn):
    path = row.path
    file_name = os.path.basename(path)
    file_type = row.file_type
    text = row.content
    chunks = row.chunks
    embeddings = row.embeddings

    document_id = conn.execute(
        "INSERT INTO documents (file_name, file_type, content) VALUES (%s, %s, %s) RETURNING id;",
        (file_name, file_type, text)
    ).fetchone()[0]

    conn.execute(
        "INSERT INTO metadata (document_id, file_path) VALUES (%s, %s) RETURNING id;", (
            document_id, path)
    )

    # COPY chunks
    with conn.cursor().copy("COPY chunks (document_id, chunk_text, chunk_vector) FROM STDIN") as copy:
        for chunk, embedding in zip(chunks, embeddings):
            copy.write_row((document_id, chunk, embedding))

    conn.commit()


MODEL_SBERT_768 = 'sentence-transformers/all-mpnet-base-v2'
MODEL_SBERT_384 = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_SPACY = "en_core_web_sm"
nlp = spacy.load(MODEL_SPACY)
model = SentenceTransformer(MODEL_SBERT_768)


def parse_file_map(path):
    try:
        content = parse_file(path)
        return Row(path=path, content=content["content"], file_type=content["metadata"].get("Content-Type"))
    except:
        return Row(path=path, content=None, file_type=None)


@pandas_udf(ArrayType(StringType()))
def clean_text_udf(documents: pd.Series) -> pd.Series:
    nlp = nlp_b.value
    docs = nlp.pipe(documents)

    def clean_text(doc):
        sentences, vectors = extract_sentences(doc)
        chunks = chunk_text_spacy(sentences, vectors)
        return chunks

    return pd.Series([clean_text(doc) for doc in docs])


@pandas_udf(ArrayType(ArrayType(FloatType())))
def encode_chunks_udf(documents: pd.Series) -> pd.Series:
    model = model_b.value
    embeddings = [model.encode(chunks, batch_size=16).tolist()
                  for chunks in documents]
    return pd.Series(embeddings)


# PATH = "/Users/anassaeed/code/nlp/GenAI/SemanticSearch/src"
PATH = "/Users/anassaeed/Downloads/TEST"

conf = SparkConf()

spark = SparkSession.builder \
    .config(conf=conf) \
    .appName("Text Extraction") \
    .master("local[*]") \
    .getOrCreate()
sc = spark.sparkContext

nlp_b = sc.broadcast(nlp)
model_b = sc.broadcast(model)

files = get_files(PATH)


def pipeline(files):
    df = sc.parallelize(files)
    # First we need to get the text from the files
    df = df.map(lambda path: parse_file_map(path)).toDF(
        ["path", "content", "file_type"])
    df = df.filter(df.content.isNotNull())
    # Then we need to clean and chunk the text
    df = df.withColumn("chunks", clean_text_udf(df.content))
    # Then we need to encode the chunks
    df = df.withColumn("embeddings", encode_chunks_udf(df.chunks))
    # Then we need to write the data to the database
    # df = df.withColumn("document_id", write_to_db_udf(df.path, df.file_type, df.content, df.chunks, df.embeddings))
    return df


df = pipeline(files)


def write_to_db_map(rows):
    with psycopg.connect(conninfo) as conn:
        for row in rows:
            write_to_db(row, conn)


df.foreachPartition(write_to_db_map)

df.show()
db.close()