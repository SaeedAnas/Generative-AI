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
from pyspark.sql.functions import udf
from pyspark.sql.types import *

from svlearn.common import SVError
from svlearn.compute import BootcampComputeJob

from svlearn.config.configuration import ConfigurationMixin
from svlearn.utils.compute_utils import _get_connection


def persist_chunks(rows) :
    """
    This method is called for each partition of the incoming dataframe, to
    persist the chunked text into the database.
    :param rows: the rows in the partition, each representing a document chunk.
    """
    # Create a MySQL connection within the partition
    connection = _get_connection()
    cursor = connection.cursor()
    mixin = ConfigurationMixin()
    config = mixin.load_config()
    clean_chunk_url = config['services']['clean_chunk']
    text_strings = []
    _ids = []
    for row in rows:
        text_strings.append(row.TEXT)
        _ids.append(row.ID)

    chunks_list = (call_chunker_endpoint(
        clean_chunk_url=clean_chunk_url, text_input=text_strings))

    for i in range(len(_ids)) :       
        doc_id = _ids[i]
        delete_query = (
            f'''
            DELETE FROM CHUNK
            WHERE
            DOC_ID = {doc_id}
            '''
        )
        cursor.execute(delete_query)
    
    connection.commit()

    cursor = connection.cursor()
    for i in range(len(_ids)):
        doc_id = _ids[i]
        update_query = (
            f'''
            UPDATE DOCUMENT set CHUNKED = {True}
            where ID = {doc_id}
            '''
        )
        cursor.execute(update_query)

        for chunk in chunks_list[i]: 
            # Suggestion from Praveen: Change from below to the parameterized approachh
            # given subsequently

            # insert_query = (
            #     f'''
            #     INSERT INTO CHUNK 
            #     (DOC_ID, TEXT, VECTORIZED, ANN_INDEXED, ES_INDEXED, VECTOR) 
            #     VALUES 
            #     ({doc_id}, '{chunk}', {False}, {False}, {False}, '' )
            #     '''
            # )
            # cursor.execute(insert_query)

            # Modified approach given by Praveen

            insert_query = '''
                INSERT INTO CHUNK
                (DOC_ID, TEXT, VECTORIZED, ANN_INDEXED, ES_INDEXED, VECTOR)
                VALUES
                (%s, %s, %s, %s, %s, %s)
            '''
            values = (doc_id, chunk, False, False, False, '')
            cursor.execute(insert_query, values)

    # Commit and close the MySQL connection for the partition
    connection.commit()
    connection.close()

def call_chunker_endpoint(clean_chunk_url: str, text_input: [str]) -> [[str]]:
    """Calls the REST endpoint taking a POST request with a list of text input that returns list of
    cleaned list of text chunks.

    Args:
        :param clean_chunk_url:
        :param text_input:
    Returns:
        A List of text output from the REST endpoint.

    """

    # Set the REST endpoint URL
    endpoint_url = clean_chunk_url

    # Create the POST request body
    request_body = {
        "text": text_input
    }

    # Make the POST request
    response = requests.post(endpoint_url, json=request_body)

    # Check the response status code
    if response.status_code == 200:
        # The request was successful
        return response.json()["chunks"]
    else:
        # The request failed
        raise SVError("Failed to call REST endpoint: {}".format(response.status_code))


class TextChunkerJob(BootcampComputeJob):
    """
    This class is the entry point for the text chunker job.
    Given the table of unchunked documents,
    it chunks each of them and persists into another table in the db.

    """

    def __init__(self):
        super().__init__(job_name='TextChunkerJob')
        self.clean_chunk_url = self.config['services']['clean_chunk']
        logging.info(f'Initializing {self.job_name} job')   

    def run(self) -> None:
        """
        This method is the entry point for the compute job where
        the documents are retrieved from DOCUMENT table, unchunked text is chunked,
        and the chunked documents stored in CHUNK table.  Also update
        CHUNKED column to True for all rows of DOCUMENT table at the end.
        :return: None
        """
        logging.info(f'Running {self.job_name} job')
        unchunked_df = self._get_unchunked_documents()
        logging.info(f'Chunking text from {unchunked_df.count()} documents')

        # below chunks as well as persists/updates
        self._chunk_text(unchunked_df)

    def _chunk_text(self, unchunked_df: DataFrame) -> None :
        """
        Chunks text from each text field in the incoming DataFrame and persists
        :param unchunked_df: DataFrame containing the list of unchunked documents
        :return: None
        """

        unchunked_df.foreachPartition(
            lambda rows : persist_chunks(rows))

    def _get_unchunked_documents(self) -> DataFrame:
        """
        Get all the unchunked documents into a DataFrame
        :return: DataFrame containing the list of unchunked documents
        """

        # Read the data from the MySQL table
        df = self._read(table="DOCUMENT")

        # Filter the data to only include rows where the chunked column is false
        df = df.filter(False == df.CHUNKED)

        # Select the id, text columns
        df = df.select("ID", "TEXT")

        # Display the dataframe
        df.show()

        return df

    def describe(self):
        return f'Chunks text from unchunked documents, and stores it in a database table'


if __name__ == '__main__':
    job = TextChunkerJob()
    job.run()
    job.spark.stop()
