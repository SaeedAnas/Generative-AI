from pydantic import BaseModel


class Config(BaseModel):
    DATA_DIR: str = "data"
    
    POSTGRES_
    
config = Config()