import asyncpg
from pgvector import HalfVector, SparseVector, Vector
from pgvector.asyncpg import register_vector
import pytest

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestAsyncpg:
    async def setup_connection(self):
        conn = await asyncpg.connect(database='pgvector_python_test')
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await register_vector(conn)
        return conn

    @pytest.mark.asyncio
    async def test_vector(self):
        conn = await self.setup_connection();
        await conn.execute('DROP TABLE IF EXISTS asyncpg_items')
        await conn.execute('CREATE TABLE asyncpg_items (id bigserial PRIMARY KEY, embedding vector(3))')

        embedding = Vector([1.5, 2, 3])
        embedding2 = [4.5, 5, 6]
        embedding3 = np.array([7.5, 8, 9]) if NUMPY_AVAILABLE else [7.5, 8, 9]
        embedding4 = None
        await conn.execute("INSERT INTO asyncpg_items (embedding) VALUES ($1), ($2), ($3), ($4)", embedding, embedding2, embedding3, embedding4)

        res = await conn.fetch("SELECT * FROM asyncpg_items ORDER BY id")
        assert res[0]['embedding'] == embedding
        assert res[1]['embedding'] == Vector(embedding2)
        assert res[2]['embedding'] == Vector(embedding3)
        assert res[3]['embedding'] is None

        # ensures binary format is correct
        text_res = await conn.fetch("SELECT embedding::text FROM asyncpg_items ORDER BY id LIMIT 1")
        assert text_res[0]['embedding'] == '[1.5,2,3]'

        await conn.close()

    @pytest.mark.asyncio
    async def test_halfvec(self):
        conn = await self.setup_connection();
        await conn.execute('DROP TABLE IF EXISTS asyncpg_items')
        await conn.execute('CREATE TABLE asyncpg_items (id bigserial PRIMARY KEY, embedding halfvec(3))')

        embedding = HalfVector([1.5, 2, 3])
        embedding2 = [4.5, 5, 6]
        embedding3 = None
        await conn.execute("INSERT INTO asyncpg_items (embedding) VALUES ($1), ($2), ($3)", embedding, embedding2, embedding3)

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
        conn = await self.setup_connection();
        await conn.execute('DROP TABLE IF EXISTS asyncpg_items')
        await conn.execute('CREATE TABLE asyncpg_items (id bigserial PRIMARY KEY, embedding bit(3))')

        embedding = asyncpg.BitString('101')  # type: ignore
        embedding2 = None
        await conn.execute("INSERT INTO asyncpg_items (embedding) VALUES ($1), ($2)", embedding, embedding2)

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
        conn = await self.setup_connection();
        await conn.execute('DROP TABLE IF EXISTS asyncpg_items')
        await conn.execute('CREATE TABLE asyncpg_items (id bigserial PRIMARY KEY, embedding sparsevec(3))')

        embedding = SparseVector([1.5, 2, 3])
        embedding2 = None
        await conn.execute("INSERT INTO asyncpg_items (embedding) VALUES ($1), ($2)", embedding, embedding2)

        res = await conn.fetch("SELECT * FROM asyncpg_items ORDER BY id")
        assert res[0]['embedding'] == embedding
        assert res[1]['embedding'] is None

        # ensures binary format is correct
        text_res = await conn.fetch("SELECT embedding::text FROM asyncpg_items ORDER BY id LIMIT 1")
        assert text_res[0]['embedding'] == '{1:1.5,2:2,3:3}/3'

        await conn.close()

    @pytest.mark.asyncio
    async def test_vector_array(self):
        conn = await self.setup_connection();
        await conn.execute('DROP TABLE IF EXISTS asyncpg_items')
        await conn.execute('CREATE TABLE asyncpg_items (id bigserial PRIMARY KEY, embeddings vector[])')

        embeddings = [Vector([1.5, 2, 3]), Vector([4.5, 5, 6])]
        await conn.execute("INSERT INTO asyncpg_items (embeddings) VALUES ($1)", embeddings)

        embeddings2 = [[1.5, 2, 3], [4.5, 5, 6]]
        await conn.execute("INSERT INTO asyncpg_items (embeddings) VALUES (ARRAY[$1, $2]::vector[])", embeddings2[0], embeddings2[1])

        if NUMPY_AVAILABLE:
            embeddings3 = [np.array([1.5, 2, 3]), np.array([4.5, 5, 6])]
            await conn.execute("INSERT INTO asyncpg_items (embeddings) VALUES (ARRAY[$1, $2]::vector[])", embeddings3[0], embeddings3[1])

        res = await conn.fetch("SELECT * FROM asyncpg_items ORDER BY id")
        assert res[0]['embeddings'] == embeddings
        assert res[1]['embeddings'] == [Vector(e) for e in embeddings2]
        if NUMPY_AVAILABLE:
            assert res[2]['embeddings'] == [Vector(e) for e in embeddings3]

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
            embedding2 = [1.5, 2, 3]
            embedding3 = None
            await conn.execute("INSERT INTO asyncpg_items (embedding) VALUES ($1), ($2), ($3)", embedding, embedding2, embedding3)

            res = await conn.fetch("SELECT * FROM asyncpg_items ORDER BY id")
            assert res[0]['embedding'] == embedding
            assert res[1]['embedding'] == Vector(embedding2)
            assert res[2]['embedding'] is None
