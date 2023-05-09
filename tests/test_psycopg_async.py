import psycopg
import numpy as np
from pgvector.psycopg import async_register_vector
import pytest


class TestAsyncPsycopg:
    @pytest.mark.asyncio
    async def test_works(self):
        conn = await psycopg.AsyncConnection.connect(database='pgvector_python_test', autocommit=True)
        await async_register_vector(conn)
        async with conn.cursor() as cur:
            await cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            await cur.execute('DROP TABLE IF EXISTS item')
            await cur.execute('CREATE TABLE item (id bigserial primary key, embedding vector(3))')


            embedding = np.array([1.5, 2, 3])
            
            await cur.execute(f"INSERT INTO item (embedding) VALUES ('{embedding.tolist()}'), (NULL)")
            await cur.execute("SELECT * FROM item ORDER BY id")
            res = await cur.fetchall()
            assert res[0][0] == 1
            assert res[1][0] == 2
            assert np.array_equal(res[0][1], embedding)
            assert res[0][1].dtype == np.float32
            assert res[1][1] is None

            # ensures binary format is correct
            await cur.execute("SELECT embedding::text FROM item ORDER BY id LIMIT 1")
            text_res = await cur.fetchall()
            assert text_res[0][0] == '[1.5,2,3]'

        await conn.close()
