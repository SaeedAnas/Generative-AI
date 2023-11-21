from psycopg.rows import class_row

from asifbot import config
from asifbot.core.text.text_embedder import TextEmbedder
from asifbot.core.db import Postgres, Qdrant
from asifbot.schema.fastapi.rag import Chunk

class RagService:
    def __init__(self):
        self.db = Postgres()
        self.qdrant = Qdrant()
        self.embedder = TextEmbedder(config.LLM.query_embedder)
        
    def get_chunks_from_db(self, ids: list[int]) -> list[str]:
        """ Function returns list of chunks from database
        Args:
            ids (list[int]): list of ids of chunks
        Returns:
            list[str]: list of chunks
        """
        with self.db.connect() as conn:
            cur = conn.cursor(row_factory=class_row(Chunk))
            cur = cur.execute(f"SELECT id, chunk_text FROM chunks WHERE id = ANY(%s)", (ids,))
            chunks = cur.fetchall()
            conn.commit()
            return chunks

    def get_n_chunks(self, text: str, n: int = 10) -> list[str]:
        """ Function returns list of n chunks of text
        Args:
            text (str): text to be chunked
            n (int): number of chunks
        Returns:
            list[str]: list of n chunks
        """
        embedding = self.embedder.embed([text])[0]
        ids = self.qdrant.search(embedding, limit=n, ids_only=True)
        chunks = self.get_chunks_from_db(ids)
        print(chunks)

        return chunks