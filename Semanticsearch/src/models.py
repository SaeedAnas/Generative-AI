from pydantic import BaseModel

class SearchResult(BaseModel):
    id: int
    document_id: int
    chunk_text: str
    score: float = 1.0
    
    def from_es_hit(hit):
        return SearchResult(
            id=hit["_source"]["id"],
            document_id=hit["_source"]["document_id"],
            text=hit["_source"]["chunk_text"],
            score=hit["_score"]
        )
        
class SearchQuery(BaseModel):
    text: str
    top_k: int = 10
    search_type: str = "text"