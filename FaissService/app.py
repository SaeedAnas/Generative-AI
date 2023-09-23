import asyncio

from pydantic import BaseModel
from fastapi import FastAPI

import faiss_handler
import kafka_consumer
from settings import settings
import logging


class Query(BaseModel):
    vector: list[float]
    top_k: int = 10


class Response(BaseModel):
    ids: list[int]


app = FastAPI()

# Create the faiss index
index = faiss_handler.create_or_load_index(settings.FAISS_INDEX_PATH)
# Create the faiss service
faiss_service = faiss_handler.FaissService(index)

# Create the kafka consumer
loop = asyncio.get_event_loop()


@app.on_event("startup")
async def startup_event():
    logging.info("Starting Application")
    consumer = kafka_consumer.KafkaConsumer(
        settings.KAFKA_BROKER_URL,
        settings.KAFKA_GROUP_ID,
        settings.KAFKA_TOPIC,
        faiss_service,
        loop
    )
    loop.create_task(consumer.consume())


@app.on_event("shutdown")
async def shutdown_event():
    await faiss_service.save(settings.FAISS_INDEX_PATH)
    logging.info("Stopping Application")

@app.get("/query", response_model=Response)
async def query(query: Query):
    results = await faiss_service.search([query.vector], query.top_k)
    return Response(ids=results[0].tolist())


@app.get("/_count")
async def count():
    return await faiss_service.count()
