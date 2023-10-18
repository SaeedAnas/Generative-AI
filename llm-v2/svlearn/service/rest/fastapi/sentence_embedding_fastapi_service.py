#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2023.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  #
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  #
#   Author: Chandar L
#  -------------------------------------------------------------------------------------------------
#
import logging as log

from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel
from typing import List

from svlearn.config.configuration import ConfigurationMixin
from svlearn.utils.compute_utils import get_port
from svlearn.text.sentence_encoder import SentenceEncoder

class InputRequest(BaseModel):
    sentences: List[str]

class SentenceEncoderModel:
    """
    This class is the entry point for the sentence encoder service.
    """
    def __init__(self):
        super().__init__()

    def initialize_embedder(self):
        self.encoder = SentenceEncoder()

    def __call__(self, request: InputRequest):
        """
        This method is called when the service is invoked,
         and it responds with embedding vectors corresponding to the sentences.
        :param request: the request object, with the sentences to be encoded
        :return: the embedding vectors, as a list inside a json object
        """
        log.info(f"Received request")
        sentences = request.sentences
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

app = FastAPI()

dispatcher = SentenceEncoderModel()

@app.post("/embedding")
async def embedding(request: InputRequest) :
    return dispatcher.__call__(request=request)

if __name__ == "__main__":
    import uvicorn
    dispatcher.initialize_embedder()

    mixin = ConfigurationMixin()
    config = mixin.load_config()
    url = config['services']['sentence_vectorizer']
    port = get_port(url) 
    
    uvicorn.run(app, host="localhost", port=port)
    log.info(f"Started serving SentenceEncoderModel")

