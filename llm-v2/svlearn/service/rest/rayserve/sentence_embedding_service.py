#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2023.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  #
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  #
#   Author: Asif Qamar
#  -------------------------------------------------------------------------------------------------
#
import logging as log

import ray
from ray import serve
from starlette.requests import Request

from svlearn.config.configuration import ConfigurationMixin
from svlearn.utils.compute_utils import get_port
from svlearn.text.sentence_encoder import SentenceEncoder


@serve.deployment(
    # specify the number of GPU's available; zero if it is run on cpu
    ray_actor_options={"num_gpus": 0},
    # the number of instances of the  deployment in the cluster
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
    # the concurrency of the deployment
    max_concurrent_queries=100,
)
class SentenceEncoderModel:
    """
    This class is the entry point for the sentence encoder service.
    """
    def __init__(self):
        super().__init__()
        self.encoder = SentenceEncoder()

    async def __call__(self, request: Request):
        """
        This method is called when the service is invoked,
         and it responds with embedding vectors corresponding to the sentences.
        :param request: the request object, with the sentences to be encoded
        :return: the embedding vectors, as a list inside a json object
        """
        log.info(f"Received request: {request}")
        payload = await request.json()
        sentences: list[str] = payload['sentences']
        try:
            log.info(f"Encoding sentences: {len(sentences)}")
            log.debug(f"Sentences: {sentences}")
            vectors = self.encoder.encode(sentences)
            log.info(f"Returning vectors: {len(vectors)}")
            vector_strings = []
            for vector in vectors:
                 floatVector = [tensor.item() for tensor in vector]
                 floatVectorString = "[" + ", ".join(map(str, floatVector)) + "]"
                 vector_strings.append(floatVectorString)           
            return {'vectors': vector_strings}
        except Exception as e:
            log.error(f'Error while encoding sentences: {e}. The sentences were: {sentences}')
            raise e


if __name__ == '__main__':
    ray.init(address='ray://localhost:10001')
    
    mixin = ConfigurationMixin()
    config = mixin.load_config()
    url = config['services']['sentence_vectorizer']
    port = get_port(url)   

    entrypoint = SentenceEncoderModel.bind()  # bind() comes from the decorator
    serve.run(entrypoint, port=port, host="0.0.0.0", route_prefix="/embedding")
    log.info(f"Started serving SentenceEncoderModel")
