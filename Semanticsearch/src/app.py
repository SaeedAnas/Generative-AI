from fastapi import FastAPI, HTTPException, Body, Depends, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from SSearch3_searcher import search

app = FastAPI()

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 template instance
templates = Jinja2Templates(directory="templates")

@app.get("/")
def read_root(request: Request):
    # Serve the search.html template when accessing the root URL
    return templates.TemplateResponse("search.html", {"request": request})

@app.post("/search")
def search_endpoint(request: Request, query: str = Body(...), top_k: int = 10, rerank_k: int = 5):
    try:
        results = search(query, top_k, rerank_k)
        # Return the results to the same search GUI along with the entered query
        return templates.TemplateResponse("search.html", {"request": request, "results": results, "query": query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
