#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2023.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: Asif Qamar
#  -------------------------------------------------------------------------------------------------
#

import logging
import os
from abc import ABC, abstractmethod

import pymysql
from pyspark.pandas import DataFrame as psdf
from pyspark.sql import DataFrame, SparkSession

from svlearn.common import SVError
from svlearn.config.configuration import ConfigurationMixin


class BootcampComputeJob(ABC, ConfigurationMixin):
    """
    Base class for all py-spark compute jobs.
    """

    def __init__(self, job_name: str = None, config_file: str = None):
        """
        Constructor
        :param job_name: the name of the compute-job in the pipeline.
        :param config_file: the name of the configuration file.
        """
        super().__init__()
        self.job_name = job_name
        self.config_file = config_file
        self.config = self.load_config(config_file)


        # Spark-related initialization
        spark_home: str = self.config['spark']['spark-home']
        spark_python_path: str = self.config['spark']['python-path']
        os.environ['SPARK_HOME'] = spark_home
        os.environ['HADOOP_HOME'] = spark_home
        os.environ['PYSPARK_DRIVER_PYTHON'] = spark_python_path
        os.environ['PYSPARK_PYTHON'] = spark_python_path

        if spark_home is None:
            raise SVError("SPARK_HOME environment variable is not set. "
                          "Please set it to the Spark installation directory.")

        # Add the spark directory to the path
        jar_file = self.config['database']['jdbc']['jar-file']
        self.driver_class = self.config['database']['jdbc']['driver-class']

        # The configuration file must specify the jar file 
        # to be used for the jdbc driver. This is ensured
        # by the configuration schema.

        jar_path = os.path.join(spark_home, "jars", jar_file)
        self.spark = (SparkSession.builder.appName(self.job_name)
                      .config("spark.jars", jar_path)
                      .getOrCreate())
        pyspark_log = logging.getLogger('pyspark')
        pyspark_log.setLevel(logging.ERROR)
        logging.getLogger("py4j").setLevel(logging.ERROR)

        # Relational database related initialization
        self.server = self.config['database']['server']
        self.port = self.config['database']['port']
        self.schema = self.config['database']['schema']
        self.user = self.config['database']['username']
        self.password = self.config['database']['password']
        self.url_format = self.config['database']['jdbc']['url-format']
        self.url = self.url_format.format(server=self.server, port=self.port, schema=self.schema)

    @abstractmethod
    def run(self):
        """
        This method is the entry point for the compute job.
        It is an abstract method in this base clas, for each of the compute jobs
        to implement.
        :return:   None
        """
        pass

    @abstractmethod
    def describe(self):
        pass

    def __repr__(self) -> str:
        """
        This method returns the name of the job, and a description.
        :return:
        """
        return super().__repr__() + f'Job_name: {self.job_name} \n {self.describe()}'

    def _add_connection_options(self, object) :
        """
        Add db connection params to the object (relevant for spark
        read/write)
        :param self:
        :param object: Could be a dataframe or spark context
        :return object: adds the object decorated with relevant params  
        """
        return (object
            .option('url', self.url) \
            .option('user', self.user) \
            .option("password", self.password) \
            .option("driver", self.driver_class))
    
    def _read(self, table: str) -> DataFrame:
            """
            Reads data from a database table using Spark.

            :param table: the name of the table to read from
            :type table: str
            :return: the DataFrame containing the data
            """
            logging.info(f'Reading from database table: {table}')
            options = self.spark.read.format('jdbc')
            return (self._add_connection_options( 
                                          options)
                                          .option('dbtable', table)
                                          .load())

    def _persist(self, df: DataFrame,
                 table: str,
                 write_mode='append',
                 truncate='false',
                 ) -> None:
        """
        Persists the DataFrame to the database.

        :param df: the DataFrame to be persisted
        :type df: DataFrame
        :param table: the name of the table to write to
        :type table: str
        :param write_mode: the write mode to use, defaults to 'append'
        :type write_mode: str
        :param truncate: whether to truncate the table before writing, defaults to 'false'
        :type truncate: str
        :return: None
        """
        writer = df.write.mode(write_mode)
        if write_mode == 'overwrite':
            writer = writer.option("truncate", f"{truncate}")

        logging.info(f'Writing to database table: {table}')
        self._add_connection_options(writer.format("jdbc")) \
            .option("dbtable", table) \
            .save()
        logging.info(f'Wrote {df.count()} rows to database table: {table}')

    def _get_connection(self) -> pymysql.Connection :
        """
        Returns a connection to the mysql-instance of the relational schema.
        :return: a connection to the mysql-instance of the relational schema.
        """
        return pymysql.connect(host=self.server,
                        port=self.port,
                        user=self.user,
                        password=self.password,
                        database=self.schema)