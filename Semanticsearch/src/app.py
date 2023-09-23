from typing import List
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
import uvicorn
from src.models import SearchResult, SearchQuery
from src.ssearch import ssearch

app = FastAPI()

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root_endpoint():
    return FileResponse("static/search.html")


@app.get("/search", response_model=List[SearchResult])
# def search(query: SearchQuery):
def search(text: str, top_k: int = 10, search_type: str = "text"):
    # results = ssearch(query.text, query.top_k)
    results = ssearch(text, top_k)
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
