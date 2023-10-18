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
from typing import List

from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import Tensor
from svlearn.common.svexception import SVError

from svlearn.config.configuration import ConfigurationMixin


class SentenceEncoder(ConfigurationMixin):
    """"
    This class provides the functionality of encoding text into vectors.
    """

    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.model_name = self.config['models']['multilingual-sentence-encoder']
        self.sentence_encoder = SentenceTransformer(self.model_name)
        self.device = self.config['models']['device']
        self.normalize_embeddings = self.config['models']['sentence-embedding']['normalize-embeddings']
        log.debug(f'Loaded sentence encoder model: {self.model_name}')
        log.debug(f'Using device: {self.device}')
        log.debug(f'Normalize embeddings: {self.normalize_embeddings}')

    def encode(self, sentences: List[str]) -> list[Tensor] | ndarray | Tensor:
        """
        This method encodes a list of texts into a list of vectors.
        :param sentences: the list of texts to be encoded
        :return: the list of vectors
        """
        try:
            log.info(f'Encoding {len(sentences)} sentences')
            
            vectors = self.sentence_encoder.encode(sentences=sentences,
                                                   device=self.device,
                                                   normalize_embeddings=True)
            log.info(f'Encoded {len(vectors)} vectors')
            return vectors
        except Exception as e:
            raise SVError(f'Error while encoding sentences: {e}')
