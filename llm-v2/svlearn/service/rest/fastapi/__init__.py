""" 
This module provides a simple FastAPI based 
service for some of the bootcamp exercises.

Recall from our discussion that you should prefer
a model-server such as Ray Serve, PyTorch Serve,
TensorFlow Serving, etc.  for production -- they
have AI-model serving specific features built 
toward concurrency control, load balancing, etc.

However, sometimes you may want to quickly test 
out your ideas, and FastAPI provides a quick route
during the development process.  This module
provides a few simple services that you can use
for testing out your ideas.
"""

from .faiss_index_builder_fastapi_service import *
from .clean_chunk_fastapi_service import *
from .sentence_embedding_fastapi_service import *
from .search_fastapi_service import *
from .query_results_reranking_fastapi_service import *
from .faiss_search_fastapi_service import *


