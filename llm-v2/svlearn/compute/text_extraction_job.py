# -------------------------------------------------------------------------------------------------
#  Copyright (c) 2023.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar
# -------------------------------------------------------------------------------------------------

import logging
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql.types import StringType

from svlearn.compute import BootcampComputeJob
from svlearn.text import TextExtraction

class TextExtractionJob(BootcampComputeJob):
    """
    This class is the entry point for the text extraction job.
    Given a directory of documents, it read all the files in the directory,
    and all the subdirectories recursively, and extracts plain text from each file.

    It then stores the extracted text in a database table.
    """

    def __init__(self):
        super().__init__(job_name='TextExtractionJob')
        logging.info(f'Initializing {self.job_name} job')
        self.text_struc = StructType([
            StructField("path", StringType(), True),
            StructField("text", StringType(), True),
            StructField("doctype", StringType(), True),
            StructField("language", StringType(), True),
            StructField("uuid", StringType(), True)
        ])

    @staticmethod
    def _udf_text_extraction(path):
        """
        A function that extracts text, its document-type and language
         from a file, given its path.
        """
        extraction = TextExtraction(path)
        return {"path": path,
                "text": extraction.text_,
                "doctype": extraction.doctype_,
                "language": extraction.language_,
                "uuid": extraction.id_
                }

    def run(self) -> None:
        """
        This method is the entry point for the compute job where
        the text is extracted from the documents, and stored in a database table.
        :return: None
        """
        logging.info(f'Running {self.job_name} job')
        files_df = self._list_documents()
        logging.info(f'Extracting text from {files_df.count()} files')
        df = self._extract_text(files_df)

        self._persist(df=df, table='DOCUMENT')

    def _extract_text(self, files_df: DataFrame) -> DataFrame:
        """
        Extracts plain-text from each file in the DataFrame
        :param files_df: DataFrame containing the list of files
        :return: DataFrame containing the extracted text
        """
        # Step 1: Extract text from each file
        files_df = files_df.withColumn('extract',
                                       udf(self._udf_text_extraction,
                                           self.text_struc)(files_df.value))
        # Step 2: Extract the columns from the nested structure
        df = files_df.select('extract.language',
                             'extract.path',
                             'extract.doctype',
                             'extract.text',
                             'extract.uuid')
        # Step 3: Rename the columns
        df = df.withColumnRenamed("language", "LANGUAGE") \
            .withColumnRenamed("uuid", "UUID") \
            .withColumnRenamed("path", "PATH") \
            .withColumnRenamed("doctype", "DOCTYPE") \
            .withColumnRenamed("text", "TEXT")

        # Step 4: Add boolean columns that help in later processing
        df = df.withColumn('CHUNKED', lit(False))

        # Step 5: Show the DataFrame
        df.show(truncate=True)
        return df

    def _list_documents(self) -> DataFrame:
        """
        Lists all the files in the directory, and returns as a DataFrame
        :return: DataFrame containing the list of files
        """
        # Step 1: List all files in the directory using pathlib
        all_files = list(Path(self.config['documents']['source-dir']).glob('*'))
        # Step 2: Read all file-names into a Spark DataFrame
        files = [str(file) for file in all_files]
        files_df = self.spark.createDataFrame(files, StringType())
        files_df.show(truncate=False)
        return files_df

    def describe(self):
        return f'Extracts text from documents in a directory, and stores it in a database table'


if __name__ == '__main__':
    job = TextExtractionJob()
    job.run()
    job.spark.stop()
