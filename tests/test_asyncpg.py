import asyncio
import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
import pytest


class TestAsyncpg:
    @pytest.mark.asyncio
    async def test_works(self):
        conn = await asyncpg.connect(database='pgvector_python_test')
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await conn.execute('DROP TABLE IF EXISTS item')
        await conn.execute('CREATE TABLE item (id bigserial PRIMARY KEY, embedding vector(3))')

        await register_vector(conn)

        embedding = np.array([1.5, 2, 3])
        await conn.execute("INSERT INTO item (embedding) VALUES ($1), (NULL)", embedding)

        res = await conn.fetch("SELECT * FROM item ORDER BY id")
        assert res[0]['id'] == 1
        assert res[1]['id'] == 2
        assert np.array_equal(res[0]['embedding'], embedding)
        assert res[0]['embedding'].dtype == np.float32
        assert res[1]['embedding'] is None

        # ensures binary format is correct
        text_res = await conn.fetch("SELECT embedding::text FROM item ORDER BY id LIMIT 1")
        assert text_res[0]['embedding'] == '[1.5,2,3]'

        await conn.close()

    @pytest.mark.asyncio
    async def test_pool(self):
        async def init(conn):
            await register_vector(conn)

        pool = await asyncpg.create_pool(database='pgvector_python_test', init=init)

        async with pool.acquire() as conn:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            await conn.execute('DROP TABLE IF EXISTS item')
            await conn.execute('CREATE TABLE item (id bigserial PRIMARY KEY, embedding vector(3))')

            embedding = np.array([1.5, 2, 3])
            await conn.execute("INSERT INTO item (embedding) VALUES ($1), (NULL)", embedding)

            res = await conn.fetch("SELECT * FROM item ORDER BY id")
            assert res[0]['id'] == 1
            assert res[1]['id'] == 2
            assert np.array_equal(res[0]['embedding'], embedding)
            assert res[0]['embedding'].dtype == np.float32
            assert res[1]['embedding'] is None
