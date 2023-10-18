# -------------------------------------------------------------------------------------------------
#  Copyright (c) 2023.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar/Chandar L
# -------------------------------------------------------------------------------------------------

import json
import logging

import requests
from pyspark.sql import DataFrame
from pyspark.sql.functions import *

from svlearn.common import SVError
from svlearn.compute import BootcampComputeJob
from svlearn.config.configuration import ConfigurationMixin
from svlearn.utils.compute_utils import _get_connection


def index_per_partition(records):
    """
    This method is called for each partition of the incoming dataframe. Each record
    represents a document chunk, and its vector is sent to the ANN indexer service.
    :param records: the rows in the partition, each representing a document chunk and its vector.
    
    """
    mixin = ConfigurationMixin()
    config = mixin.load_config()
    ann_index_url = config['services']['faiss_index_builder']
    input_vectors: List[Tuple[int, List[float]]] = []
    _ids = []
    dimension = -1
    for row in records:
        json_array = json.loads(row.VECTOR)
        float_vector: List[float] = [float(x) for x in json_array]
        if dimension == -1 :
            dimension = len(float_vector)
        tuple:Tuple[int, List[float]]  = (row.ID, float_vector)
        input_vectors.append(tuple)
        _ids.append(row.ID)


    call_faiss_indexer_endpoint(ann_index_url, input_vectors)

    connection = _get_connection()
    cursor = connection.cursor()

    for _id in _ids:
        update_query = (
            f'''
            UPDATE CHUNK set ANN_INDEXED = {True}
            where ID = {_id}
            '''
        )
        cursor.execute(update_query)

    connection.commit()
    connection.close()

def call_faiss_indexer_endpoint(ann_index_url: str, input_vectors: List[Tuple[int, List[float]]]):
    """
    Calls the REST endpoint taking a POST request with a vector 
    input that calls ANN indexer service.

    Args:
        :param ann_index_url: the URL of the ANN indexer service
        :param input_vectors: the list of vectors to be indexed

    """

    # Set the REST endpoint URL
    endpoint_url = ann_index_url

    # Create the POST request body
    request_body = {
        "vectors": input_vectors
    }

    # Make the POST request
    response = requests.post(endpoint_url, json=request_body)

    # Check the response status code
    if response.status_code == 200:
        # The request was successful
        return
    else:
        # The request failed
        raise SVError("Failed to call REST endpoint: {}".format(response.status_code))


class ANNIndexerJob(BootcampComputeJob):
    """
    This class is the entry point for the ANN Indexer job.
    Given the table of un-indexed vectors,
    it will send the documents to ANN index for indexing.

    """

    def __init__(self):
        super().__init__(job_name='ANNIndexerJob')
        logging.info(f'Initializing {self.job_name} job')

    def run(self) -> None:
        """
        This method is the entry point for the compute job where
        the vectors are retrieved from CHUNK table, un-ann-indexed vectors are sent to ANN index.
        Also update ANN_INDEXED column to True for all rows of CHUNK table at the end.
        :return: None
        """
        logging.info(f'Running {self.job_name} job')
        unindexed_df = self._get_un_ann_indexed_documents()
        logging.info(f'populating ann index from {unindexed_df.count()} vectors')
        self._ann_index_text(unindexed_df)

    def _ann_index_text(self, unindexed_df: DataFrame):
        """
        Sends vectors from each vector field in the incoming DataFrame to ANN index
        Also updates ANN_INDEXED to True once it posts to the index
        :param unindexed_df: DataFrame containing the list of unindexed documents
        """
        # for each partition of incoming dataframe, call the rest endpoint to faiss index
        unindexed_df.foreachPartition(lambda partition: index_per_partition(partition))

    def _get_un_ann_indexed_documents(self) -> DataFrame:
        """
        Get all the un-es-indexed documents into a DataFrame
        :return: DataFrame containing the list of unindexed documents
        """

        # Read the data from the MySQL table
        df = self._read(table="CHUNK")

        # Filter the data to only include rows where the es-indexed column is false
        df = df.filter(False == df.ANN_INDEXED)

        # Select the id, vector columns
        df = df.select("ID", "VECTOR")

        # Display the dataframe
        df.show()

        return df

    def describe(self):
        return f'Indexes vectors from unindexed vectors'


if __name__ == '__main__':
    job = ANNIndexerJob()
    job.run()
    job.spark.stop()
