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
import re

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

from svlearn.config.configuration import ConfigurationMixin


class ChunkText(ConfigurationMixin):
    """
     This class provides the functionality of chunking text into smaller pieces,
      after some cleanup. The cleanup includes removing newlines,
      tabs, and extra spaces within a sentence, etc.
    """

    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        model_name = self.config['models']['spacy-model']
        self.sentence_encoder_model = self.config['models']['spacy-sentence-embedding-model']
        self.nlp = spacy.load(model_name)
        self.sentence_encoder = SentenceTransformer(self.sentence_encoder_model)
        self.chunk_size = self.config['text']['chunk-size']
        self.similarity_threshold = self.config['text']['chunk-similarity-threshold']

    @staticmethod
    def cosine_similarity(α, β):
        dot_product = np.dot(α, β)
        norm_α = np.linalg.norm(α)
        norm_β = np.linalg.norm(β)
        return dot_product / (norm_α * norm_β)
    
    def create_list_of_chunks(self, text_list: list[str]) -> list[list[str]]:
        output_chunks = []
        for text in text_list:
            output_chunks.append(self.create_chunks(text=text))
        return output_chunks

    def create_chunks(self, text: str) -> list[str]:
        """
        This method takes a text, cleans it a bit and returns a list of chunks.
        :param text: the text to be chunked
        :return: the chunks as a list
        """
        doc = self.nlp(text)
        # Step 1: Clean the text
        cleaned_text = []
        for ind, sentence in enumerate(doc.sents):
            cleaned = re.sub('\s+', ' ', sentence.text).strip()
            cleaned = re.sub('\n+', ' ', cleaned).strip()
            cleaned_text.append(cleaned) if (len(cleaned) > 0) else None
        # Step 2: Chunk it to pieces
        chunks = []
        text = ''
        current_length = 0
        previous_sentence = None
        for sentence in cleaned_text:
            tokens = len(self.nlp(sentence))
            current_sentence = self.sentence_encoder.encode(sentence)
            similarity = 1
            if previous_sentence is not None:
                similarity = self.cosine_similarity(previous_sentence, current_sentence)
            if (current_length + tokens < self.chunk_size and
                    (previous_sentence is None
                     or similarity > self.similarity_threshold)):
                text += ' ' + sentence
                current_length += tokens

            else:
                text = text.strip()
                chunks.append(text) if (len(text) > 0) else None
                text = sentence
                current_length = tokens

            # update the previous sentence
            previous_sentence = current_sentence

        chunks.append(text) if (len(text) > 0) else None
        return chunks

# ASIF: important to do this:
"""
#nlp = spacy.load(MODEL_SPACY, disable=["ner","tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
It will remove unnessary components and make it faster.
"""
