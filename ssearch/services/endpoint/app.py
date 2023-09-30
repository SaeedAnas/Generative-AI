from typing import List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from typing import List
from fastapi.responses import FileResponse
import uvicorn

from ssearch.core.models import Results, SearchResult
from ssearch.core.search import search_es, search_vector, search_hybrid
import time

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8001",
    "http://localhost:4200"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root_endpoint():
    return FileResponse("static/search.html")


@app.get("/search", response_model=Results)
def search(text: str, top_k: int = 10, search_type: str = "text"):
    start = time.time()
    results = search_hybrid(text, top_k)
    duration = time.time() - start
    return Results(results=results, time=duration)


@app.post("/search", response_model=Results)
def searchp(text: str, top_k: int = 10, search_type: str = "text"):
    start = time.time()
    results = search_hybrid(text, top_k)
    duration = time.time() - start
    return Results(results=results, time=duration)


@app.get("/search_vector", response_model=List[SearchResult])
def search_vector(text: str, top_k: int = 10):
    results = search_vector(text, top_k)
    return results


@app.get("/search_es", response_model=List[SearchResult])
def search_elasticsearch(text: str, top_k: int = 10):
    results = search_es(text, top_k)
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
