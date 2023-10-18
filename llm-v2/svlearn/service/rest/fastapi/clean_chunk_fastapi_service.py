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

from svlearn.config.configuration import ConfigurationMixin
from svlearn.utils.compute_utils import get_port
from svlearn.text.text_chunker import ChunkText

from pydantic import BaseModel
from typing import List

class InputRequest(BaseModel):
    text: List[str]

class CleanChunkModel:
    """
    This class is the entry point for the clean/chunk service.
    """
    def __init__(self):
        super().__init__()


    def initialize_chunker(self):
        self.chunker = ChunkText()

    def __call__(self, request: InputRequest):
        """
        This method is called when the service is invoked, and it returns the chunks of the text.
        :param request: the request object, with the text to be chunked
        :return: the chunks of the text, as a list inside a json object
        """
        log.info(f"Received request")
        text = request.text
        chunks = self.chunker.create_list_of_chunks(text_list=text)
        log.info(f"Returning chunks: {len(chunks)}")
        return {'chunks': chunks}

app = FastAPI()

dispatcher = CleanChunkModel()

@app.post("/chunker")
async def chunk(request: InputRequest) :
    return dispatcher.__call__(request=request)


if __name__ == "__main__":
    import uvicorn

    dispatcher.initialize_chunker()

    mixin = ConfigurationMixin()
    config = mixin.load_config()
    url = config['services']['clean_chunk']
    port = get_port(url)  

    uvicorn.run(app, host="localhost", port=port)
    log.info(f"Started serving CleanChunkModel")
