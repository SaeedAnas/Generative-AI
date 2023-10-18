from typing import List, Tuple
from pydantic import BaseModel

class RerankerInputRequest(BaseModel):
    query: str
    results: List[Tuple[int,str]]