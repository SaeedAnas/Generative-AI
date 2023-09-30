from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()


class ImagePath(BaseModel):
    image_path: str


class Caption(BaseModel):
    caption: str


@serve.deployment(
    # ray_actor_options={"num_cpus": 12, "num_gpus": 0},
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
    max_concurrent_queries=10,
)
@serve.ingress(app)
class ImageCaptioner:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base")

    @serve.batch(max_batch_size=32)
    async def batch_caption(self, image_paths: List[str]) -> List[str]:
        raw_images = [Image.open(path).convert('RGB') for path in image_paths]
        inputs = self.processor(raw_images, return_tensors="pt", padding=True)
        out = self.model.generate(**inputs)
        return self.processor.batch_decode(out, skip_special_tokens=True)

    @app.get("/caption")
    async def caption(self, request: ImagePath) -> Caption:
        caption = await self.batch_caption(request.image_path)
        return Caption(caption=caption)


serve_app = ImageCaptioner.bind()

if __name__ == "__main__":
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base")

    IMAGE_PATH = '/Users/anassaeed/Downloads/F7IDWtjakAAOnam.jpg'
    start = time.time()
    images = [
        IMAGE_PATH,
        IMAGE_PATH,
        IMAGE_PATH,
    ]
    raw_images = [Image.open(img).convert('RGB') for img in images]
    inputs = processor(raw_images, return_tensors="pt", padding=True)
    out = model.generate(**inputs)
    print(processor.batch_decode(out, skip_special_tokens=True))
    print('Time elapsed: ', time.time() - start)
