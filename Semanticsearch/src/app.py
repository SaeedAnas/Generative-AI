from typing import Optional
from typing import Any, Dict, AnyStr, List, Union
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
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SearchResult(BaseModel):
    text:str
    score: float

# Serve static files from the 'static' directory
#app.mount("/static", StaticFiles(directory="/Users/praveen/dev/project-SV/Assignment1/github/Generative-AI/Semanticsearch/src/static"), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root_endpoint():
    return FileResponse("static/search.html")

@app.post("/search2")
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


@app.post("/search")
async def get_body(request: Dict[str,Any]): 
#   {'query': keyString, 'search_type': sType, 'top_k': top, 'rerank_k': rerank},

    try:
        req = request
        if req['search_type'] not in ["all","text", "images", "audio", "video"]:
            raise ValueError(f"Invalid search type: {req['search_type']}")
        results = search(req['query'], req['top_k'], req['rerank_k'],search_type="text") #req.search_type
       
        # Create the required response structure and return as JSON
        response_data = {'result': results}
        return response_data
      
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return await request.json()

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
