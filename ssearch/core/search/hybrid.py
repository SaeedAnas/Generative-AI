import logging

from sentence_transformers import CrossEncoder
from psycopg.rows import class_row

from ssearch.core.db.postgres import db
from ssearch.core.models import SearchResult, DocumentMetadata
from ssearch.core.search.es import search_es
from ssearch.core.search.vector import search_faiss, fetch_chunk_ids

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')


def fetch_document_metadata(ids):
    try:
        with db.connect() as conn:
            cur = conn.cursor(row_factory=class_row(DocumentMetadata))
            cur.execute(
                "SELECT id, file_name, file_type FROM documents WHERE id = ANY(%s)", [ids])
            results = cur.fetchall()
            conn.commit()
            return results
    except Exception as e:
        logging.error(f"Error fetching document metadata: {e}")
        return []


def rerank(query, results: SearchResult):
    pairs = [(query, hit.chunk_text) for hit in results]
    scores = cross_encoder.predict(pairs)

    for result, score in zip(results, scores):
        result.score = score

    results.sort(key=lambda x: x.score, reverse=True)
    return results


def merge_results(es_results, faiss_results):
    # Merge the results from ES and FAISS
    # Ensure that the results are unique
    results = list(
        {result.id: result for result in es_results + faiss_results}.values())
    return results


def search_hybrid(query, top_k):
    es_results = search_es(query, top_k)
    ids = search_faiss(query, top_k)
    faiss_results = fetch_chunk_ids(ids)
    results = merge_results(es_results, faiss_results)
    results = rerank(query, results)
    document_ids = [result.document_id for result in results]
    document_metadata = fetch_document_metadata(document_ids)
    metadata_dict = {metadata.id: metadata for metadata in document_metadata}
    for result in results:
        result.metadata = metadata_dict[result.document_id]

    return results


if __name__ == "__main__":
    # Test the search function
    query = "What is diffusion?"
    top_k = 10
    results = search_hybrid(query, top_k)
    print(results)
    # es_results = search_es(query, top_k)
    # print(es_results)
    # ids = search_faiss(query, top_k)
    # faiss_results = fetch_chunk_ids(ids)
    # print(faiss_results)
    # results = es_results + faiss_results
    # results = rerank(query, results)
    # print(results)
