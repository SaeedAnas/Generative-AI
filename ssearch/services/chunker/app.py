# MAD RESPECT TO THIS DUDE
# https://medium.com/@npolovinkin/how-to-chunk-text-into-paragraphs-using-python-8ae66be38ea6

from ray import serve

from fastapi import FastAPI
import asyncio

import spacy
from sentence_transformers import SentenceTransformer

import numpy as np

from pydantic import BaseModel
from typing import List

import time

import ssearch.services.chunker.util as util


MODEL = 'all-MiniLM-L6-v2'

app = FastAPI()


def load_spacy():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


class Text(BaseModel):
    text: str


class Chunks(BaseModel):
    chunks: List[str]


@serve.deployment(
    ray_actor_options={"num_cpus": 12, "num_gpus": 0},
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
    max_concurrent_queries=100,
)
@serve.ingress(app)
class TextChunkingService:
    def __init__(self):
        self.model = SentenceTransformer(MODEL)
        self.nlp = load_spacy()

    @serve.batch()
    async def sentencize(self, docs: List[str]) -> List[List[str]]:
        docs = self.nlp.pipe(docs, batch_size=16)
        return [util.sentencize(doc) for doc in docs]

    @serve.batch(max_batch_size=800, batch_wait_timeout_s=0.1)
    async def embed_sentences(self, sentences: List[str]) -> List[np.ndarray]:
        embeddings = self.model.encode(
            sentences, batch_size=16, convert_to_numpy=True)
        return embeddings

    @app.get("/chunk")
    async def chunk(self, text: Text) -> Chunks:
        sentences = await self.sentencize(text.text)
        embeddings = await asyncio.gather(*[self.embed_sentences(sentence)
                                            for sentence in sentences])
        embeddings = np.stack(embeddings)
        split_points = util.get_split_points(embeddings)
        chunks = util.get_chunks(sentences, split_points)
        return Chunks(chunks=chunks)


serve_app = TextChunkingService.bind()
