from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging as _log
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from svlearn.config import ConfigurationMixin
from svlearn.utils.compute_utils import get_port
from svlearn.service.rest.fastapi.search_fastapi_service  import HybridSearch 
from svlearn.compute.image_extraction_job import text_query

dispatcher = HybridSearch()
dispatcher.initialize()

mixin = ConfigurationMixin()
config = mixin.load_config()
image_dir = config['documents']['image-dir']
    
app = FastAPI()
app.mount("/static", StaticFiles(directory=image_dir), name="static")

origins = ["*"]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)
  
class SearchRequest(BaseModel):
    query: str
    search_type: str
    top_k: int = 10  # optional with a default value of 10

@app.post("/search")
async def get_body(req: SearchRequest): 
    try:
        if req.search_type not in ["all", "text", "image", "audio", "video"]:
            raise ValueError(f"Invalid search type: {req.search_type}")
        if req.search_type == "text":
            search_results = dispatcher.hybrid_search(query=req.query, k=req.top_k)
            results = transform_output(search_results)
        elif req.search_type == "image":
            results = text_query(req.query,req.top_k,'image')            
        elif req.search_type == "audio":
            results = dispatcher.audio_search()
        else: 
            results = {}
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def transform_output(result):
    reranked_chunks = result.get('neighbours', [])
    output_json = [{"score": "{:.6f}".format(item[2]), "text": item[1]} for item in reranked_chunks]
    return {"result": output_json}

if __name__ == "__main__":
    import uvicorn
    dispatcher.initialize()
    
    url = config['services']['search']
    port = get_port(url)  

    uvicorn.run(app, host="localhost", port=port)
    _log.info(f"Started serving HybridSearch")