from pydantic import BaseModel

class RAGRequest(BaseModel):
    text: str
    n: int = 10
    
class Chunk(BaseModel):
    id: int
    chunk_text: str
    
class RAGResponse(BaseModel):
    chunks: list[Chunk]