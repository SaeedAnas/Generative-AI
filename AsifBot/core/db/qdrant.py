from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from asifbot import config

class Qdrant:
    def __init__(self):
        self.client = QdrantClient(config.HOST, port=config.QDRANT_PORT)
        self.collection_name = config.QDRANT_COLLECTION
        self.embedding_dim = config.LLM.embedding_dim
        
    def create_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )
        
    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)
        
    def upsert(self, points, ids):
        return self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[
                PointStruct(id=id, vector=point.tolist()) for id, point in zip(ids, points)
            ]
        )
        
    def search(self, point, limit=10, ids_only=False):
        points = self.client.search(
            collection_name=self.collection_name,
            query_vector=point,
            limit=limit
        )
        
        if ids_only:
            return [point.id for point in points]
        else:
            return points
        
    def info(self):
        return self.client.get_collection(collection_name=self.collection_name)
