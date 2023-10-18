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

import faiss
import numpy as np

from svlearn.common import file_exists
from svlearn.common.svexception import SVError
from svlearn.config import ConfigurationMixin

log.basicConfig(level=log.DEBUG)

#
# TODO: We need to train a few of the indexers before using them. 
# For example, the IVF indexer needs to be trained with a set of vectors.
#
# We need to add a method to train the indexers.
#

class FaissIndexer(ConfigurationMixin):
    """
    This class wraps the Faiss library,
    and keeps track of the index. It can be used to add vectors to the index, and
    ensure the updated index is persisted to disk.
    """

    def __init__(self,
                 index_file: str = None,
                 index_type: str = 'brute_force',
                 dimension: int = 64,
                 metric_type: int = faiss.METRIC_L2,
                 nlist: int = 100):
        """
        Constructor for the FaissIndexer class. 
        """
        super().__init__()

        if index_file is not None and file_exists(index_file):
            log.info(f'Loading existing Faiss-index from file: {index_file}')
            self.index = self.load_index(index_file)
            log.info(f'Index loaded with dimension: {self.index.d} and size: {self.index.ntotal}')

        else:
            log.info(f'Creating new Faiss-index of type: {index_type}')
            if index_type == 'brute_force':
                log.info(f'Creating new Faiss-index')
                self.index = self.create_brute_force_index(dimension=dimension)
            elif index_type == 'hnsw':
                log.info(f'Creating new Faiss-index')
                self.index = self.create_hnsw_index(dimension=dimension)
            elif index_type == 'ivf':
                log.info(f'Creating new Faiss-index')
                self.index = self.create_ivf_index(dimension=dimension, metric_type = metric_type, nlist=nlist)
            else:
                log.error("Invalid index type. Valid types are: brute_force, hnsw, ivf."
                          "We are still cooking the other types. Please check back later.")
                raise ValueError("Invalid index type. Valid types are: brute_force, hnsw, ivf.")

            log.info(f'Index created with dimension: {self.index.d} and size: {self.index.ntotal}')
            # wrap up with IndexIDMap
            self.index = faiss.IndexIDMap(self.index)

    def create_brute_force_index(self, dimension: int = 64):
        """
        Create a brute-force index. Beware that this index can be very slow for large datasets.
        """
        log.info(f'Creating brute-force index with dimension: {dimension}')
        index = faiss.IndexFlatL2(dimension)
        log.info(f'Index created with dimension: {dimension} and size: {index.ntotal}')
        return index

    def create_hnsw_index(self,
                          dimension: int = 64,
                          M: int = 16,
                          efConstruction: int = 40):

        """
        Create a HNSW index. This index is much faster than the brute-force index; however,
        it is not as accurate as the brute-force index.

        :param dimension: the dimension of the vectors
        :param M: the number of neighbors to explore at each query point
        :param efConstruction: the number of neighbors to index at construction time
        """
        log.info(f'Creating HNSW index with dimension: {dimension}, M: {M}, efConstruction: {efConstruction}')     
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = efConstruction
        log.info(f'Index created with dimension: {dimension} and size: {index.ntotal}')
        return index

    def load_index(self, index_file: str):
        """
        Load an existing index from file.
        """
        log.info(f'Loading existing Faiss-index from file: {index_file}')
        if not file_exists(index_file) :
            raise SVError(f'File does not exist: {index_file}. Error: {e}')

        try:
            index = faiss.read_index(index_file)
        except Exception as e:
            raise SVError(f'Error while loading Faiss-index from file: {index_file}. Error: {e}')

        log.info(f'Index loaded from {index_file}')
        return index

    def create_ivf_index(self, 
                        dimension: int = 64, 
                        metric_type: int = faiss.METRIC_L2, 
                        nlist: int = 100):
        """
        Create an IVF index. This index is much faster than the brute-force index; 
        however, it needs to be trained with a set of vectors before it can be used for search.

        :param dimension: the dimension of the vectors
        :param metric_type: the type of metric to use for computing distances between vectors
        :param nlist: the number of cells in the quantizer 
        """
        
        log.info(f'Creating IVF index with dimension: {dimension}, metric_type: {metric_type}, nlist: {nlist}')
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric_type)
        log.info(f'Index created with dimension: {dimension} and size: {index.ntotal}')
        return index

    def save_index(self, index_file: str):
        """
        Save the index to file.

        :param index_file: the name of the file to save the index to.
        """
        log.info(f'Saving Faiss-index to file: {index_file}')
        try:
            faiss.write_index(self.index, index_file)
        except Exception as e:
            raise SVError(f'Error while saving Faiss-index to file: {index_file}. Error: {e}')

        log.info(f'Faiss−Index saved to file: {index_file}')

    def size(self):
        return self.index.ntotal

    def add(self, vector_list=[(str, [float])]) -> [(str, str)]:
        current = self.index.ntotal
        log.info(f'Adding {len(vector_list)} vectors to the index')
        print(vector_list)
        ids = [id for (id, _) in vector_list]
        vectors = [np.array(vector) for _, vector in vector_list]
        self.index.add_with_ids(np.array(vectors), ids)
