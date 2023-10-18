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
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from svlearn.config.configuration import ConfigurationMixin


class QueryResultsReRanker(ConfigurationMixin):
    """
     This class provides the functionality of chunking text into smaller pieces,
      after some cleanup. The cleanup includes removing newlines,
      tabs, and extra spaces within a sentence, etc.
    """

    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.model_name = self.config['models']['multilingual-cross-encoder']
        log.info(f'Configuration specifies the cross encoder model: '
                 f'{self.model_name}. Loading it...')
        try:
            self.model = CrossEncoder(self.model_name)
            log.info(f'Loaded cross encoder model: {self.model_name}')
        except Exception as e:
            log.error(f'Error while loading cross encoder model: {e}. '
                      f'Check the model name in the config file.')
            raise e

    def rerank(self, query: str, results: List[Tuple[int, str]]) -> List[Tuple[int, str, float]]:
        """
        This method takes a query and a list of results, and returns a re-ranked
        list of results reverse-sorted by the scores determined by the model.
        :param query: the query text
        :param results: the list of results and their scores.
        """
        log.info(f'Request for re-ranking the results for query: {query} and {len(results)} results')
        log.debug(f'Results to re-rank: {results}, given query: {query}')

        if query is None or len(query) == 0:
            log.error(f'Query cannot be empty: {query}')
            raise ValueError('Query cannot be empty')
        if results is None or len(results) == 0:
            log.error(f'Candidate results cannot be empty: {results}')
            raise ValueError('Candidates cannot be empty')
        try:
            # Use the model to compute scores for each result in relation to the query
            scores = self.model.predict([[query, result[1]] for result in results])
            # Sort the results by their scores
            sorted_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
            sorted_result_tuples = [(result[0][0], result[0][1], float(result[1])) for result in sorted_results]
            return sorted_result_tuples
        except Exception as e:
            log.error(f'Model inference error while reranking results: {e}. '
                      f'The query was: {query}. The results were: {results}')
            raise e


if __name__ == '__main__':

    rr = QueryResultsReRanker()

    query = "What is the structure of an atom?"
    results = [

        (1, """Quack-quack go the ducks"""),
        (2, """It was the best of times, it was the worst of times, it was the age of wisdom,
                    it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity,
                    it was the season of Light, it was the season of Darkness, it was the spring of hope,
                    it was the winter of despair, we had everything before us, we had nothing before us,
                    we were all going direct to Heaven, we were all going direct the other way – in short,
                    the period was so far like the present period, that some of its noisiest authorities
                    insisted on its being received, for good or for evil, in the superlative degree of comparison only."""),

        (3, """The quick brown fox jumped over the lazy dog"""),

        (4, """This code is well-structured and easy to read"""),

        (5, """The atom has an orbital structure, 
                       with the atom's electrons orbiting about a nucleus of protons and neutrons. """),

        (6, """Quantum mechanics studies the behavior of atoms, and its constituent particles"""),

        (7, """The atom is the smallest unit of matter that can't be broken down using any chemical means"""),
    ]

    re_ranked_results = rr.rerank(query, results)
    for result in re_ranked_results:
        print(f'{result}')
