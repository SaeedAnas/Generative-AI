# Semantic Search Engine

## Introduction
In this project, we have constructed a hybrid semantic search engine utilizing both traditional keyword-based indexing (via Elasticsearch) and semantic vector-based searching (through Faiss). The core idea behind semantic search is that it uses embeddings to represent the meanings of words or chunks of text. When querying, it compares the semantic similarity (or distance) between the query's embedding and the embeddings of stored documents, allowing for more intuitive and contextually relevant search results.

## Document Processing
The first phase involved processing raw documents to be indexed:
- **Tika-Python Integration**: Leveraged Apache Tika with Python to extract text from various document types.
- **Text Cleaning**: After extracting raw text, we pre-processed this content to remove any HTML tags, non-alphanumeric characters, convert to lowercase, and eliminate any extra spaces.
- **Text Chunking**: Instead of indexing entire documents, we split the content into smaller, more manageable chunks. This approach enhances the granularity of the search and allows users to pinpoint specific sections of documents that are most relevant to their queries.

## 5 Levels Of Text Splitting/Chunking:
1. **Character Split**: The most basic form of splitting, which divides text based on individual characters.
2. **Recursive Character Text Splitter**: An advanced version of character splitting, where recursion is employed to split at various character levels.
3. **Document Specific Chunker**: This technique is tailored for specific document types, whether it's Code, Markdown, or PDF.
4. **Semantic Chunking (Embeddings Only)**: 
    - This is a more advanced chunking technique where embeddings of sequential text chunks are examined.
    - We check the distance (or similarity) between embeddings. If the distance is significant, a chunk cut-off is made.
    - A cosine similarity threshold is established. If the features of two chunks fall below this threshold in similarity, they are considered distinct.
5. **Reasoned Chunking (Agent-like)**: This involves a system that iterates through the text and intelligently decides the boundaries of a chunk.

## Indexing
After processing and chunking the documents, the next phase was to index them:
- **Vector Embeddings**: Used `SentenceTransformer` to convert chunks of text into dense vector embeddings.
- **FAISS Indexing**: Indexed these embeddings using Faiss, a library specialized for efficient similarity search and clustering of dense vectors.
- **Elasticsearch Indexing**: Alongside Faiss, we also indexed chunks in Elasticsearch, a popular open-source search and analytics engine.

## Hybrid Search
The search phase involves querying both the Faiss and Elasticsearch indices:
- **Query Transformation**: Transformed the user's query into a vector using `SentenceTransformer`.
- **Elasticsearch Querying**: Performed a keyword-based search on Elasticsearch.
- **FAISS Vector Search**: Used the query's vector representation to perform a semantic similarity search in the Faiss index.
- **Results Aggregation**: Aggregated results from both sources.
- **Re-ranking with Cross-Encoder**: Used a more powerful cross-encoder model from `SentenceTransformer` to re-rank the aggregated results based on semantic similarity to the query.

## Conclusion
This project showcases the power of combining traditional keyword search with modern semantic search techniques. It provides a foundation for building more advanced, intuitive search engines that can understand the context and nuances of user queries.
