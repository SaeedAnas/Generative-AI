from ray import serve

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from imagebind import data
import torch
import numpy as np
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from typing import Any
import os

data.BPE_PATH = "ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz"

app = FastAPI()


class Request(BaseModel):
    data: str


def validate_path(path: str) -> bool:
    return os.path.exists(path)


@serve.deployment(
    # ray_actor_options={"num_cpus": 12, "num_gpus": 0},
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
    max_concurrent_queries=100,
)
@serve.ingress(app)
class ImageBindEmbedding:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

    @serve.batch()
    async def embed(self, paths: list[tuple[Any, str]]) -> list[tuple[Any, np.ndarray]]:
        inputs = {}
        mapping = {}
        for idx, p in enumerate(paths):
            modality, path = p
            if modality not in inputs:
                inputs[modality] = []
                mapping[modality] = []
            inputs[modality].append(path)
            mapping[modality].append(idx)

        for modality in inputs:
            if modality == ModalityType.TEXT:
                inputs[modality] = data.load_and_transform_text(
                    inputs[modality], self.device)
            elif modality == ModalityType.VISION:
                inputs[modality] = data.load_and_transform_vision_data(
                    inputs[modality], self.device)
            elif modality == ModalityType.AUDIO:
                inputs[modality] = data.load_and_transform_audio_data(
                    inputs[modality], self.device)
            else:
                raise Exception("Invalid modality type")

        with torch.no_grad():
            embeddings = self.model(inputs)

        outputs = [None] * len(paths)
        for (modality, idxs) in mapping.items():
            for idx, embedding in zip(idxs, embeddings[modality]):
                outputs[idx] = (modality, embedding)

        return outputs

    @app.get("/text")
    async def embed_text(self, text: Request) -> list[float]:
        modality, embedding = await self.embed((ModalityType.TEXT, text.data))
        return embedding.tolist()

    @app.get("/image")
    async def embed_image(self, path: Request) -> list[float]:
        if not self.check_path(path.data):
            return HTTPException(status_code=404, detail="File not found")

        modality, embedding = await self.embed((ModalityType.VISION, path.data))
        return embedding.tolist()

    @app.get("/audio")
    async def embed_audio(self, path: Request) -> list[float]:
        if not self.check_path(path.data):
            return HTTPException(status_code=404, detail="File not found")

        modality, embedding = await self.embed((ModalityType.AUDIO, path.data))
        return embedding.tolist()


serve_app = ImageBindEmbedding.bind()
