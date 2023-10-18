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
import numpy as np

from svlearn.text.faiss_indexer import FaissIndexer

if __name__ == '__main__':
    a = (1, [1, 2, 3])
    b = (4, [4, 5, 6])
    c = (7, [7, 8, 9])
    d = (8, [10, 11, 12])
    e = (32, [13, 14, 15])

    input = [a, b, c, d, e]
    ids = np.array([id for id, _ in input])
    vectors = [np.array(vector) for _, vector in input]
    data = np.vstack(vectors).astype('float32')
    dimension = 3  # vector dimension

    print('=' * 100)
    nlist = min(len(data), 100)
    faiss_index = FaissIndexer(dimension=3, index_type='ivf', nlist=nlist)
    print(f'Index initialized with dimension: {dimension} and size: {faiss_index.size()}')

    if not faiss_index.index.is_trained :
        faiss_index.index.train(data)

    faiss_index.index.add_with_ids(data, ids)
    print(data.shape)
    print(faiss_index.size())
 
