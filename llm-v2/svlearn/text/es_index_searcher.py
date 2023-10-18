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
from elasticsearch import Elasticsearch

from svlearn.config import ConfigurationMixin


class ElasticSearchIndex(ConfigurationMixin):
    """

    """
    def __init__(self):
        super().__init__()
        self.server = self.config['elasticsearch']['server']
        self.port = self.config['elasticsearch']['port']
        self.index_name = self.config['elasticsearch']['index_name']
        self.index = Elasticsearch[{'host': self.server, 'port': self.port}]

    def search(self, query_str) -> [(str, float)]:
        """
        This method searches the elastic-search index for the given query string.

        :param query_str: the query string
        :return: a list of tuples, where each tuple contains the document-id and the score.
        """
        query = {"match_all": {query_str}}
        self.index.search(index=self.index_name, query=query)