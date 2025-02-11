import asyncpg
import numpy as np
from pgvector import HalfVector, SparseVector, Vector
from pgvector.asyncpg import register_vector
import pytest


class TestAsyncpg:
    @pytest.mark.asyncio
    async def test_vector(self):
        conn = await asyncpg.connect(database='pgvector_python_test')
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await conn.execute('DROP TABLE IF EXISTS asyncpg_items')
        await conn.execute('CREATE TABLE asyncpg_items (id bigserial PRIMARY KEY, embedding vector(3))')

        await register_vector(conn)

        embedding = Vector([1.5, 2, 3])
        embedding2 = np.array([4.5, 5, 6])
        await conn.execute("INSERT INTO asyncpg_items (embedding) VALUES ($1), ($2), (NULL)", embedding, embedding2)

        res = await conn.fetch("SELECT * FROM asyncpg_items ORDER BY id")
        assert np.array_equal(res[0]['embedding'], embedding.to_numpy())
        assert res[0]['embedding'].dtype == np.float32
        assert np.array_equal(res[1]['embedding'], embedding2)
        assert res[2]['embedding'] is None

        # ensures binary format is correct
        text_res = await conn.fetch("SELECT embedding::text FROM asyncpg_items ORDER BY id LIMIT 1")
        assert text_res[0]['embedding'] == '[1.5,2,3]'

        await conn.close()

    @pytest.mark.asyncio
    async def test_halfvec(self):
        conn = await asyncpg.connect(database='pgvector_python_test')
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await conn.execute('DROP TABLE IF EXISTS asyncpg_items')
        await conn.execute('CREATE TABLE asyncpg_items (id bigserial PRIMARY KEY, embedding halfvec(3))')

        await register_vector(conn)

        embedding = HalfVector([1.5, 2, 3])
        embedding2 = [4.5, 5, 6]
        await conn.execute("INSERT INTO asyncpg_items (embedding) VALUES ($1), ($2), (NULL)", embedding, embedding2)

        res = await conn.fetch("SELECT * FROM asyncpg_items ORDER BY id")
        assert res[0]['embedding'] == embedding
        assert res[1]['embedding'] == HalfVector(embedding2)
        assert res[2]['embedding'] is None

        # ensures binary format is correct
        text_res = await conn.fetch("SELECT embedding::text FROM asyncpg_items ORDER BY id LIMIT 1")
        assert text_res[0]['embedding'] == '[1.5,2,3]'

        await conn.close()

    @pytest.mark.asyncio
    async def test_bit(self):
        conn = await asyncpg.connect(database='pgvector_python_test')
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await conn.execute('DROP TABLE IF EXISTS asyncpg_items')
        await conn.execute('CREATE TABLE asyncpg_items (id bigserial PRIMARY KEY, embedding bit(3))')

        await register_vector(conn)

        embedding = asyncpg.BitString('101')
        await conn.execute("INSERT INTO asyncpg_items (embedding) VALUES ($1), (NULL)", embedding)

        res = await conn.fetch("SELECT * FROM asyncpg_items ORDER BY id")
        assert res[0]['embedding'].as_string() == '101'
        assert res[0]['embedding'].to_int() == 5
        assert res[1]['embedding'] is None

        # ensures binary format is correct
        text_res = await conn.fetch("SELECT embedding::text FROM asyncpg_items ORDER BY id LIMIT 1")
        assert text_res[0]['embedding'] == '101'

        await conn.close()

    @pytest.mark.asyncio
    async def test_sparsevec(self):
        conn = await asyncpg.connect(database='pgvector_python_test')
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await conn.execute('DROP TABLE IF EXISTS asyncpg_items')
        await conn.execute('CREATE TABLE asyncpg_items (id bigserial PRIMARY KEY, embedding sparsevec(3))')

        await register_vector(conn)

        embedding = SparseVector([1.5, 2, 3])
        await conn.execute("INSERT INTO asyncpg_items (embedding) VALUES ($1), (NULL)", embedding)

        res = await conn.fetch("SELECT * FROM asyncpg_items ORDER BY id")
        assert res[0]['embedding'] == embedding
        assert res[1]['embedding'] is None

        # ensures binary format is correct
        text_res = await conn.fetch("SELECT embedding::text FROM asyncpg_items ORDER BY id LIMIT 1")
        assert text_res[0]['embedding'] == '{1:1.5,2:2,3:3}/3'

        await conn.close()

    @pytest.mark.asyncio
    async def test_vector_array(self):
        conn = await asyncpg.connect(database='pgvector_python_test')
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await conn.execute('DROP TABLE IF EXISTS asyncpg_items')
        await conn.execute('CREATE TABLE asyncpg_items (id bigserial PRIMARY KEY, embeddings vector[])')

        await register_vector(conn)

        embeddings = [Vector([1.5, 2, 3]), Vector([4.5, 5, 6])]
        await conn.execute("INSERT INTO asyncpg_items (embeddings) VALUES ($1)", embeddings)

        embeddings2 = [np.array([1.5, 2, 3]), np.array([4.5, 5, 6])]
        await conn.execute("INSERT INTO asyncpg_items (embeddings) VALUES (ARRAY[$1, $2]::vector[])", embeddings2[0], embeddings2[1])

        res = await conn.fetch("SELECT * FROM asyncpg_items ORDER BY id")
        assert np.array_equal(res[0]['embeddings'][0], embeddings[0].to_numpy())
        assert np.array_equal(res[0]['embeddings'][1], embeddings[1].to_numpy())
        assert np.array_equal(res[1]['embeddings'][0], embeddings2[0])
        assert np.array_equal(res[1]['embeddings'][1], embeddings2[1])

        await conn.close()

    @pytest.mark.asyncio
    async def test_pool(self):
        async def init(conn):
            await register_vector(conn)

        pool = await asyncpg.create_pool(database='pgvector_python_test', init=init)

        async with pool.acquire() as conn:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            await conn.execute('DROP TABLE IF EXISTS asyncpg_items')
            await conn.execute('CREATE TABLE asyncpg_items (id bigserial PRIMARY KEY, embedding vector(3))')

            embedding = Vector([1.5, 2, 3])
            embedding2 = np.array([1.5, 2, 3])
            await conn.execute("INSERT INTO asyncpg_items (embedding) VALUES ($1), ($2), (NULL)", embedding, embedding2)

            res = await conn.fetch("SELECT * FROM asyncpg_items ORDER BY id")
            assert np.array_equal(res[0]['embedding'], embedding.to_numpy())
            assert res[0]['embedding'].dtype == np.float32
            assert np.array_equal(res[1]['embedding'], embedding2)
            assert res[2]['embedding'] is None
