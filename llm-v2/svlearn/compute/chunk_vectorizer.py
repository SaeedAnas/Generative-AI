# -------------------------------------------------------------------------------------------------
#  Copyright (c) 2023.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar/Chandar L
# -------------------------------------------------------------------------------------------------

import logging

import requests
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *

from svlearn.common import SVError
from svlearn.compute import BootcampComputeJob

from svlearn.config.configuration import ConfigurationMixin
from svlearn.utils.compute_utils import _get_connection

def persist_vectors(rows):
    """
    This method is called for each partition of the incoming dataframe, to 
    persist the embedding vectors associated with each chunk into the database.
    """
    # Create a MySQL connection within the partition
    connection = _get_connection()
    cursor = connection.cursor()
    mixin = ConfigurationMixin()
    config = mixin.load_config()
    sentence_vectorizer_url = config['services']['sentence_vectorizer']    

    sentences = []
    _ids = []
    for row in rows:
        sentences.append(row.TEXT)
        _ids.append(row.ID)

    vector_strings = call_embedding_endpoint(
        sentence_vectorizer_url=sentence_vectorizer_url,
                                 text_input=sentences)

    partition_size = len(_ids)
    for i in range(partition_size):
        vector_string = vector_strings[i]
        _id = _ids[i]
        update_query = (
            f'''
            UPDATE CHUNK 
            set VECTORIZED = {True}, VECTOR = '{vector_string}'
            where ID = {_id}
            '''
        )
        cursor.execute(update_query)

    # Commit and close the MySQL connection for the partition
    connection.commit()
    connection.close()


def call_embedding_endpoint(sentence_vectorizer_url: str, text_input: [str]):
    """Calls the REST endpoint taking a POST request with a text input that returns the sentence embedding.

    Args:
        :param sentence_vectorizer_url:
        :param text_input:
    Returns:
        A vector embedding of the input text.

    """

    # Set the REST endpoint URL
    endpoint_url = sentence_vectorizer_url

    # Create the POST request body
    request_body = {
        "sentences": text_input
    }

    # Make the POST request
    response = requests.post(endpoint_url, json=request_body)

    # Check the response status code
    if response.status_code == 200:
        # The request was successful
        return response.json()["vectors"]
    else:
        # The request failed
        raise SVError("Failed to call REST endpoint: {}".format(response.status_code))


class ChunkVectorizerJob(BootcampComputeJob):
    """
    This class is the entry point for the chunk vectorizer job.
    Given the table of chunked documents,
    it vectorizes each of them and persists into another table in the db.

    """

    def __init__(self):
        super().__init__(job_name='ChunkVectorizerJob')
        self.sentence_vectorizer_url = self.config['services']['sentence_vectorizer']
        logging.info(f'Initializing {self.job_name} job')

    def run(self) -> None:
        """
        This method is the entry point for the compute job where
        the documents are retrieved from CHUNK table, un-vectorized text is vectorized,
        and the vectorized documents stored back in the CHUNK table.  Also update
        VECTORIZED column to True for all rows of CHUNK table at the end.
        :return: None
        """
        logging.info(f'Running {self.job_name} job')
        un_vectorized_df = self._get_unvectorized_chunks()
        logging.info(f'Chunking text from {un_vectorized_df.count()} chunks')
        df = self._vectorize_chunk(un_vectorized_df)
    
    def _vectorize_chunk(self, un_vectorized_df: DataFrame) -> None:
        """
        Vectorizes chunks from each chunk field in the incoming DataFrame, also updates vectors in db
        :param un_vectorized_df: DataFrame containing the list of unvectorized chunks
        :return: None
        """

        un_vectorized_df.foreachPartition(lambda partition: persist_vectors(partition))

    def _get_unvectorized_chunks(self) -> DataFrame:
        """
        Get all the unvectorized chunks into a DataFrame
        :return: DataFrame containing the list of unvectorized chunks
        """

        # Read the data from the MySQL table
        df = self._read(table="CHUNK")

        # Filter the data to only include rows where the vectorized column is false
        df = df.filter(False == df.VECTORIZED)

        # Select the id and TEXT columns
        df = df.select("ID", "TEXT")

        # Display the dataframe
        df.show()

        return df

    def describe(self):
        return f'vectorizes text from chunked documents, and stores it in a database table'


if __name__ == '__main__':
    job = ChunkVectorizerJob()
    job.run()
    job.spark.stop()
