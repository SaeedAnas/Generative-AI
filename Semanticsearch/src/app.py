from typing import List, Dict, Union
from fastapi import FastAPI, HTTPException, Body, Depends, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from SSearch3_searcher import search
from typing import Optional
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
from fastapi import Response

class SearchResult(BaseModel):
    text: str
    score: float

app = FastAPI()

STATIC_DIR="/Users/praveen/dev/project-SV/Assignment1/github/Generative-AI/Semanticsearch/src"

# Serve static files from the 'static' directory
#app.mount("/static", StaticFiles(directory="/Users/praveen/dev/project-SV/Assignment1/github/Generative-AI/Semanticsearch/src/static"), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root_endpoint():
    #return FileResponse("/Users/praveen/dev/project-SV/Assignment1/github/Generative-AI/Semanticsearch/src/static/search.html")
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)