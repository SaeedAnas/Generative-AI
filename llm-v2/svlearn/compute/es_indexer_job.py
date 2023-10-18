# -------------------------------------------------------------------------------------------------
#  Copyright (c) 2023.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar/Chandar L
# -------------------------------------------------------------------------------------------------

import logging as _log

from pyspark.sql import DataFrame
from pyspark.sql.functions import *

from svlearn.compute import BootcampComputeJob
from svlearn.utils.compute_utils import _get_connection, _get_elastic_client
    
def call_rest_partition(records):
    """
    This method is called for each partition of the incoming dataframe.
    For each partition, it does the following:

    1. Create a connection to the ElasticSearch instance.
    2. For each row in the partition, it calls the ElasticSearch REST endpoint
         to index the text.
    3. Update the CHUNK table to set the ES_INDEXED column to True.
    """
    es_client, index_name = _get_elastic_client()
    connection = _get_connection()
    cursor = connection.cursor()

    # it would be better if we used the bulk-update API for ES
    for row in records:
        doc = {"id" : row.ID, "text" : row.TEXT}
        es_client.index(index=index_name, document=doc, id=row.ID)
        update_query = (
            f'''
            UPDATE CHUNK set ES_INDEXED = {True}
            where ID = {row.ID}
            '''
        )
        cursor.execute(update_query)

    connection.commit()
    connection.close()

class ESIndexerJob(BootcampComputeJob):
    """
    This class is the entry point for the ES Indexer job.
    Given the table of un-indexed documents,
    it will send the documents to ES for indexing.

    """

    def __init__(self):
        super().__init__(job_name='ESIndexerJob')
        self.es_index_url = self.config['services']['es_index']
        _log.info(f'Initializing {self.job_name} job')

    def run(self) -> None:
        """
        This method is the entry point for the compute job where
        the documents are retrieved from DOCUMENT table, un-es-indexed text is sent to ES index.
        Also update ES_INDEXED column to True for all rows of DOCUMENT table at the end.
        :return: None
        """
        _log.info(f'Running {self.job_name} job')
        unindexed_df = self._get_un_es_indexed_documents()
        _log.info(f'Populating ES index from {unindexed_df.count()} documents')
        self._es_index_text(unindexed_df)

    def _es_index_text(self, unindexed_df: DataFrame):
        """
        Sends text from each text field in the incoming DataFrame to ES index
        :param unindexed_df: DataFrame containing the list of unindexed documents
        """

        # for each element of incoming dataframe, call the rest endpoint to ES index
        # and update document saying it is indexed
        unindexed_df.foreachPartition(lambda partition: call_rest_partition(partition))

    def _get_un_es_indexed_documents(self) -> DataFrame:
        """
        Get all the un-es-indexed documents into a DataFrame
        :return: DataFrame containing the list of unindexed documents
        """

        # Read the data from the MySQL table
        df = self._read(table="CHUNK")

        # Filter the data to only include rows where the es-indexed column is false
        df = df.filter(False == df.ES_INDEXED)

        # Select the id, text columns
        df = df.select("ID", "TEXT")

        # Display the dataframe
        df.show()

        return df

    def describe(self):
        return f'Indexes text from unindexed documents'


if __name__ == '__main__':
    job = ESIndexerJob()
    job.run()
    job.spark.stop()
