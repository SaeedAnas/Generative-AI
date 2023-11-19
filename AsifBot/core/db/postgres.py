from psycopg_pool import ConnectionPool

from asifbot import config
from asifbot.schema.sql import TABLES

class Postgres:
    def __init__(self, min_size=1, max_size=20, **kwargs):
        self.conninfo = f"dbname={config.POSTGRES_DB} user={config.POSTGRES_USER} password={config.POSTGRES_PASSWORD} host={config.POSTGRES_HOST} port={config.POSTGRES_PORT}"
        self.pool = ConnectionPool(
            conninfo=self.conninfo,
            min_size=min_size,
            max_size=max_size,
            **kwargs
        )
        
    def connect(self):
        return self.pool.connection()
    
    def close(self):
        self.pool.close()

    def create_tables(self):
        with self.connect() as conn:
            for _, path in TABLES.items():
                conn.execute(open(path, "r").read())
            conn.commit()
        
    def drop_tables(self):
        with self.connect() as conn:
            for table, _ in TABLES.items():
                conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
            conn.commit()        