The order in which the jobs in this folder need to be run are:

1. text_extraction_job.py
2. chunker_job.py
3. es_indexer_job.py
4. chunk_vectorizer.py
5. ann_indexer_job.py

Before running the above jobs, we need to make sure that the following services are running (either ray-serve or fastapi):

1. clean_chunk_service
2. elasticsearch service
3. sentence_embedding_service
4. faiss_index_builder_service

