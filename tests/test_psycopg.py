import numpy as np
from pgvector.psycopg import register_vector, register_vector_async, HalfVector, SparseVector
import psycopg
import pytest

conn = psycopg.connect(dbname='pgvector_python_test', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
conn.execute('DROP TABLE IF EXISTS psycopg_items')
conn.execute('CREATE TABLE psycopg_items (id bigserial PRIMARY KEY, embedding vector(3), half_embedding halfvec(3), binary_embedding bit(3), sparse_embedding sparsevec(3))')

register_vector(conn)


class TestPsycopg:
    def setup_method(self, test_method):
        conn.execute('DELETE FROM psycopg_items')

    def test_vector(self):
        embedding = np.array([1.5, 2, 3])
        conn.execute('INSERT INTO psycopg_items (embedding) VALUES (%s), (NULL)', (embedding,))

        res = conn.execute('SELECT * FROM psycopg_items ORDER BY id').fetchall()
        assert np.array_equal(res[0][1], embedding)
        assert res[0][1].dtype == np.float32
        assert res[1][1] is None

    def test_vector_binary_format(self):
        embedding = np.array([1.5, 2, 3])
        res = conn.execute('SELECT %b::vector', (embedding,), binary=True).fetchone()[0]
        assert np.array_equal(res, embedding)

    def test_vector_text_format(self):
        embedding = np.array([1.5, 2, 3])
        res = conn.execute('SELECT %t::vector', (embedding,)).fetchone()[0]
        assert np.array_equal(res, embedding)

    def test_vector_binary_format_correct(self):
        embedding = np.array([1.5, 2, 3])
        res = conn.execute('SELECT %b::vector::text', (embedding,)).fetchone()[0]
        assert res == '[1.5,2,3]'

    def test_vector_text_format_non_contiguous(self):
        embedding = np.flipud(np.array([1.5, 2, 3]))
        assert not embedding.data.contiguous
        res = conn.execute('SELECT %t::vector', (embedding,)).fetchone()[0]
        assert np.array_equal(res, np.array([3, 2, 1.5]))

    def test_vector_binary_format_non_contiguous(self):
        embedding = np.flipud(np.array([1.5, 2, 3]))
        assert not embedding.data.contiguous
        res = conn.execute('SELECT %b::vector', (embedding,)).fetchone()[0]
        assert np.array_equal(res, np.array([3, 2, 1.5]))

    def test_text_copy(self):
        embedding = np.array([1.5, 2, 3])
        cur = conn.cursor()
        with cur.copy("COPY psycopg_items (embedding, half_embedding, binary_embedding, sparse_embedding) FROM STDIN") as copy:
            copy.write_row([embedding, HalfVector(embedding), '101', SparseVector.from_dense(embedding)])

    def test_binary_copy(self):
        embedding = np.array([1.5, 2, 3])
        cur = conn.cursor()
        with cur.copy("COPY psycopg_items (embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.write_row([embedding])

    def test_binary_copy_set_types(self):
        embedding = np.array([1.5, 2, 3])
        cur = conn.cursor()
        with cur.copy("COPY psycopg_items (id, embedding, half_embedding, sparse_embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(['int8', 'vector', 'halfvec', 'sparsevec'])
            copy.write_row([1, embedding, HalfVector(embedding), SparseVector.from_dense(embedding)])

    def test_halfvec(self):
        conn.execute('DROP TABLE IF EXISTS half_items')
        conn.execute('CREATE TABLE half_items (id bigserial PRIMARY KEY, embedding halfvec(3))')

        embedding = HalfVector([1.5, 2, 3])
        conn.execute('INSERT INTO half_items (embedding) VALUES (%s)', (embedding,))

        res = conn.execute('SELECT * FROM half_items ORDER BY id').fetchall()

    def test_halfvec_binary_format(self):
        embedding = HalfVector([1.5, 2, 3])
        res = conn.execute('SELECT %b::halfvec', (embedding,), binary=True).fetchone()[0]
        assert res.to_list() == [1.5, 2, 3]

    def test_halfvec_text_format(self):
        embedding = HalfVector([1.5, 2, 3])
        res = conn.execute('SELECT %t::halfvec', (embedding,)).fetchone()[0]
        assert res.to_list() == [1.5, 2, 3]

    def test_sparsevec(self):
        conn.execute('DROP TABLE IF EXISTS sparse_items')
        conn.execute('CREATE TABLE sparse_items (id bigserial PRIMARY KEY, embedding sparsevec(6))')

        embedding = SparseVector.from_dense([0, 1.5, 0, 2, 0, 3])
        conn.execute('INSERT INTO sparse_items (embedding) VALUES (%s)', (embedding,))

        res = conn.execute('SELECT * FROM sparse_items ORDER BY id').fetchall()
        assert res[0][1].to_dense() == [0, 1.5, 0, 2, 0, 3]

    def test_sparsevec_binary_format(self):
        embedding = SparseVector.from_dense([1.5, 2, 3])
        res = conn.execute('SELECT %b::sparsevec', (embedding,), binary=True).fetchone()[0]
        assert res.to_dense() == [1.5, 2, 3]

    def test_sparsevec_text_format(self):
        embedding = SparseVector.from_dense([1.5, 2, 3])
        res = conn.execute('SELECT %t::sparsevec', (embedding,)).fetchone()[0]
        assert res.to_dense() == [1.5, 2, 3]

    def test_bit(self):
        res = conn.execute('SELECT %s::bit(3)', ('101',)).fetchone()[0]
        assert res == '101'

    def test_bit_binary_format(self):
        res = conn.execute('SELECT %b::bit(3)', ('101',), binary=True).fetchone()[0]
        assert res == b'\x00\x00\x00\x03\xa0'

    def test_bit_text_format(self):
        res = conn.execute('SELECT %t::bit(3)', ('101',)).fetchone()[0]
        assert res == '101'

    @pytest.mark.asyncio
    async def test_async(self):
        conn = await psycopg.AsyncConnection.connect(dbname='pgvector_python_test', autocommit=True)

        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await conn.execute('DROP TABLE IF EXISTS psycopg_items')
        await conn.execute('CREATE TABLE psycopg_items (id bigserial PRIMARY KEY, embedding vector(3))')

        await register_vector_async(conn)

        embedding = np.array([1.5, 2, 3])
        await conn.execute('INSERT INTO psycopg_items (embedding) VALUES (%s), (NULL)', (embedding,))

        async with conn.cursor() as cur:
            await cur.execute('SELECT * FROM psycopg_items ORDER BY id')
            res = await cur.fetchall()
            assert np.array_equal(res[0][1], embedding)
            assert res[0][1].dtype == np.float32
            assert res[1][1] is None
