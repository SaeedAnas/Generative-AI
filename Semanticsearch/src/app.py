from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Body, Depends, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
from models import SearchResult, SearchQuery
#from ssearch import ssearch_vector, search_es
from SSearch3_searcher import search

class SearchResult(BaseModel):
    text:str
    score: float

app = FastAPI()

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root_endpoint():
    return FileResponse("static/search.html")

@app.post("/search")
def search_endpoint(request: Request, query: str = Body(...), top_k: int = 10, rerank_k: int = 5, search_type: Optional[str] = Body("text")):
    try:
        if search_type not in ["text", "images", "audio", "video"]:
            raise ValueError(f"Invalid search type: {search_type}")
        results = search(query, top_k, rerank_k,search_type=search_type)
       
        # Create the required response structure and return as JSON
        response_data = {'result': results}
        return response_data
      
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
@app.get("/search_vector", response_model=List[SearchResult])
def search_vector(text: str, top_k: int = 10):
    results = ssearch_vector(text, top_k)
    return results

@app.get("/search_es", response_model=List[SearchResult])
def search_elasticsearch(text: str, top_k: int = 10):
    results = search_es(text, top_k)
    return results
"""
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
