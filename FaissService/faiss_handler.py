import faiss
import numpy as np

import asyncio
from settings import settings
import logging

class AsyncReadWriteLock:
    def __init__(self):
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._readers = 0
        
    async def acquire_read(self):
        await self._read_lock.acquire()
        self._readers += 1
        if self._readers == 1:
            await self._write_lock.acquire()
        self._read_lock.release()
        
    async def release_read(self):
        await self._read_lock.acquire()
        self._readers -= 1
        if self._readers == 0:
            self._write_lock.release()
        self._read_lock.release()
        
    async def acquire_write(self):
        await self._write_lock.acquire()
        
    async def release_write(self):
        self._write_lock.release()

class FaissService:
    def __init__(self, index):
        self.index = index
        self.lock = AsyncReadWriteLock()

    async def add(self, vectors, ids):
        vectors = np.asarray(vectors)
        
        if vectors.shape[1] != 384:
            logging.error(f"Invalid vectors: {vectors}")
            return

        await self.lock.acquire_write()
        self.index.add(vectors)
        await self.lock.release_write()

    async def search(self, vectors, k):
        vectors = np.asarray(vectors)
        await self.lock.acquire_read()
        _, I = self.index.search(vectors, k)
        await self.lock.release_read()
        return I
    
    async def count(self):
        await self.lock.acquire_read()
        ntotal = self.index.ntotal
        await self.lock.release_read()
        return ntotal
    
    async def save(self, path):
        await self.lock.acquire_write()
        faiss.write_index(self.index, path)
        await self.lock.release_write()
        
def create_or_load_index(path):
    try:
        index = faiss.read_index(path)
    except:
        index = faiss.IndexFlatL2(384)
    return index