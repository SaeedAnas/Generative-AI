from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    id: int
    file_name: str
    file_type: str


class SearchResult(BaseModel):
    id: int
    document_id: int
    chunk_text: str
    score: float = 1.0
    source: str = "faiss"
    metadata: DocumentMetadata = None

    def from_es_hit(hit):
        return SearchResult(
            id=hit["_source"]["id"],
            document_id=hit["_source"]["document_id"],
            chunk_text=hit["_source"]["chunk_text"],
            score=hit["_score"],
            source="es"
        )


class SearchQuery(BaseModel):
    text: str
    top_k: int = 10
    search_type: str = "text"


class SearchResults(BaseModel):
    results: list[SearchResult]
    time: float
