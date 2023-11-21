from fastapi import FastAPI

from asifbot import config
from asifbot.core.llm.rag import RagService
from asifbot.schema.fastapi.rag import RAGRequest, RAGResponse

app = FastAPI()
rag = RagService()

@app.get("/", response_model=RAGResponse)
async def get_chunks(request: RAGRequest):
    chunks = rag.get_n_chunks(request.text, request.n)
    return RAGResponse(chunks=chunks)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.ENDPOINTS.rag_endpoint)
    print("Started RAG service")