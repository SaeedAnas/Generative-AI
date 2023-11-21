from pydantic import BaseModel
from pydantic_settings import BaseSettings

class LLM(BaseModel):
    # Embedding
    chunk_embedder: str = "jinaai/jina-embedding-b-en-v1"
    query_embedder: str = "jinaai/jina-embedding-b-en-v1"
    embedding_dim: int = 768

    # LLM
    llm: str = "mistralai/Mistral-7B-v0.1"
    
class ENDPOINTS(BaseModel):
    vllm_endpoint: int = 8000
    rag_endpoint: int = 8001

class Config(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    QDRANT_PORT: int
    QDRANT_COLLECTION: str
    
    DATA_DIR: str

    HOST: str
    
    LLM: LLM = LLM()
    ENDPOINTS: ENDPOINTS = ENDPOINTS()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    
config = Config()