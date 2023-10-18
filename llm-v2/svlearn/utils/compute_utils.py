import pymysql
from svlearn.config.configuration import ConfigurationMixin
import logging as _log
from elasticsearch import Elasticsearch
from svlearn.common import SVError
from urllib.parse import urlparse

def get_port(http_url: str) -> int :
    """
    Get the port from a url
    """
    parsed_url = urlparse(http_url)

    default_port = 80
    if parsed_url.scheme == "https":
        default_port = 443

    port = parsed_url.port
    if port is None:
        port = default_port 
            
    return port

def _get_connection() :
    """
    Create a connection to the relational database.
    """
    mixin = ConfigurationMixin()
    config = mixin.load_config()
    server = config['database']['server']
    port = config['database']['port']
    schema = config['database']['schema']
    user = config['database']['username']
    password = config['database']['password']

    return pymysql.connect(host=server,
                    port=port,
                    user=user,
                    password=password,
                    database=schema)

def _get_elastic_client():
    """
    Create a connection to the ElasticSearch instance.
    """
    mixin = ConfigurationMixin()
    config = mixin.load_config()
    es_password = config['elasticsearch']['password']
    es_home = config['elasticsearch']['eshome']
    es_url = config['services']['es_index']
    es_index = config['elasticsearch']['index_name']
    
    # Create and return the client instance/index name

    _log.info(f'Creating connection to ElasticSearch instance at {es_url}')

    try:
        es =  (
            Elasticsearch(
            es_url,
            ca_certs=es_home + "/config/certs/http_ca.crt",
            basic_auth=("elastic", es_password)
            )
        )
    except Exception as e:
        _log.error(f'Error while creating connection to ElasticSearch instance: {e}')
        raise SVError(f'Error while creating connection to ElasticSearch instance: {e}')

    _log.info(f'Created connection to ElasticSearch instance at {es_url}')       

    return es, es_index