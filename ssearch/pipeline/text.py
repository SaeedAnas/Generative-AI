import psycopg
from tika import parser
import pandas as pd
from pyspark.sql.types import StringType, ArrayType, FloatType, IntegerType, Row, StructType, StructField
from pyspark.sql.functions import udf, pandas_udf, from_json
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import os
import aiohttp
import asyncio

# import findspark
# findspark.init()

conninfo = "dbname=postgres user=postgres password=example host=localhost port=5432"


def file_exists(path):
    return os.path.exists(path)


def parse_file(path):
    if not file_exists(path):
        return Row(path=path, content=None, file_type=None)

    try:
        content = parser.parse_file(path)
        return Row(path=path, content=content["content"], file_type=content["metadata"].get("Content-Type"))
    except:
        return Row(path=path, content=None, file_type=None)


@pandas_udf(ArrayType(ArrayType(StringType())))
async def chunk_text_udf(documents: pd.Series) -> pd.Series:
    url = "localhost:8000"
    async with aiohttp.ClientSession() as session:
        responses = [session.get(url, doc) for doc in documents]
        responses = asyncio.gather(*responses)
        return pd.Series([resp.json()["chunks"] for resp in responses])


@pandas_udf(ArrayType(ArrayType(FloatType())))
async def embed_chunks_udf(documents: pd.Series) -> pd.Series:
    url = "localhost:8001"
    async with aiohttp.ClientSession() as session:
        responses = [session.get(url, chunks) for chunks in documents]
        responses = asyncio.gather(*responses)
        return pd.Series([resp.json()["vector"] for resp in responses])


def write_to_db(
        path,
        file_type,
        text,
        chunks,
        embeddings,
        conn
):
    file_name = os.path.basename(path)

    document_id = conn.execute(
        "INSERT INTO documents (file_name, file_type, content) VALUES (%s, %s, %s) RETURNING id;",
        (file_name, file_type, text)
    ).fetchone()[0]

    conn.execute(
        "INSERT INTO metadata (document_id, file_path) VALUES (%s, %s) RETURNING id;",
        (document_id, path)
    )

    with conn.cursor().copy("COPY chunks (document_id, chunk_text, chunk_vector) FROM STDIN") as copy:
        for chunk, embedding in zip(chunks, embeddings):
            copy.write_row((document_id, chunk, embedding))

    conn.commit()


def write_to_db_map(rows):
    with psycopg.connect(conninfo) as conn:
        for row in rows:
            write_to_db(
                path=row.path,
                file_type=row.file_type,
                text=row.content,
                chunks=row.chunks,
                embeddings=row.embeddings,
                conn=conn
            )


conf = SparkConf()

spark = SparkSession.builder \
    .config(conf=conf) \
    .appName("Text Extraction") \
    .master("local[2]") \
    .getOrCreate()
sc = spark.sparkContext

TOPIC = "text-pipeline"

filePathSchema = StructType([
    StructField("path", StringType(), False),
])

# Subscribe to kafka topic batch
df = spark \
    .read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:29092") \
    .option("subscribe", TOPIC) \
    .load()

# df = df.selectExpr("CAST(value AS STRING)").select(
#     from_json("value", filePathSchema).alias("data")).select("data.*").toDF(["path"])
df = df.selectExpr("CAST(value AS STRIG)")

df.show()

# df = df.rdd.map(lambda path: parse_file(path)).toDF(
#     ["path", "content", "file_type"]
# )

# df = df.withColumn("chunks", chunk_text_udf(df.content))
# df = df.withColumn("embeddings", embed_chunks_udf(df.chunks))

# df.forEachPartition(write_to_db_map)
