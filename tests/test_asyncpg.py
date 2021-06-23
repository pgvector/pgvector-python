import numpy as np
from pgvector.asyncpg import register_vector
import asyncio
import asyncpg
import pytest


@pytest.mark.asyncio
async def test_works():
    conn = await asyncpg.connect(database='pgvector_python_test')
    await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    await conn.execute('DROP TABLE IF EXISTS item')
    await conn.execute('CREATE TABLE item (id bigserial primary key, factors vector(3))')

    await register_vector(conn)

    factors = np.array([1.5, 2, 3])
    await conn.execute("INSERT INTO item (factors) VALUES ($1), (NULL)", factors)

    res = await conn.fetch("SELECT * FROM item ORDER BY id")
    assert res[0]['id'] == 1
    assert res[1]['id'] == 2
    assert np.array_equal(res[0]['factors'], factors)
    assert res[0]['factors'].dtype == np.float32
    assert res[1]['factors'] is None

    # ensures binary format is correct
    text_res = await conn.fetch("SELECT factors::text FROM item ORDER BY id LIMIT 1")
    assert text_res[0]['factors'] == '[1.5,2,3]'

    await conn.close()
