import numpy as np
from pgvector import HalfVector, SparseVector, Vector
from pgvector.psycopg2 import register_vector
import psycopg2
from psycopg2.extras import DictCursor, RealDictCursor, NamedTupleCursor
from psycopg2.pool import ThreadedConnectionPool

conn = psycopg2.connect(dbname='pgvector_python_test')
conn.autocommit = True

cur = conn.cursor()
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
cur.execute('DROP TABLE IF EXISTS psycopg2_items')
cur.execute('CREATE TABLE psycopg2_items (id bigserial PRIMARY KEY, embedding vector(3), half_embedding halfvec(3), binary_embedding bit(3), sparse_embedding sparsevec(3), embeddings vector[], half_embeddings halfvec[], sparse_embeddings sparsevec[])')

register_vector(cur)


class TestPsycopg2:
    def setup_method(self):
        cur.execute('DELETE FROM psycopg2_items')

    def test_vector(self):
        embedding = np.array([1.5, 2, 3])
        cur.execute('INSERT INTO psycopg2_items (embedding) VALUES (%s), (NULL)', (embedding,))

        cur.execute('SELECT embedding FROM psycopg2_items ORDER BY id')
        res = cur.fetchall()
        assert res[0][0] == Vector(embedding)
        assert res[1][0] is None

    def test_vector_class(self):
        embedding = Vector([1.5, 2, 3])
        cur.execute('INSERT INTO psycopg2_items (embedding) VALUES (%s), (NULL)', (embedding,))

        cur.execute('SELECT embedding FROM psycopg2_items ORDER BY id')
        res = cur.fetchall()
        assert res[0][0] == embedding
        assert res[1][0] is None

    def test_halfvec(self):
        embedding = HalfVector([1.5, 2, 3])
        cur.execute('INSERT INTO psycopg2_items (half_embedding) VALUES (%s), (NULL)', (embedding,))

        cur.execute('SELECT half_embedding FROM psycopg2_items ORDER BY id')
        res = cur.fetchall()
        assert res[0][0] == embedding
        assert res[1][0] is None

    def test_bit(self):
        embedding = '101'
        cur.execute('INSERT INTO psycopg2_items (binary_embedding) VALUES (%s), (NULL)', (embedding,))

        cur.execute('SELECT binary_embedding FROM psycopg2_items ORDER BY id')
        res = cur.fetchall()
        assert res[0][0] == embedding
        assert res[1][0] is None

    def test_sparsevec(self):
        embedding = SparseVector([1.5, 2, 3])
        cur.execute('INSERT INTO psycopg2_items (sparse_embedding) VALUES (%s), (NULL)', (embedding,))

        cur.execute('SELECT sparse_embedding FROM psycopg2_items ORDER BY id')
        res = cur.fetchall()
        assert res[0][0] == embedding
        assert res[1][0] is None

    def test_vector_array(self):
        embeddings = [Vector([1.5, 2, 3]), Vector([4.5, 5, 6])]
        cur.execute('INSERT INTO psycopg2_items (embeddings) VALUES (%s::vector[])', (embeddings,))

        cur.execute('SELECT embeddings FROM psycopg2_items ORDER BY id')
        res = cur.fetchone()
        assert res[0][0] == embeddings[0]
        assert res[0][1] == embeddings[1]

    def test_halfvec_array(self):
        embeddings = [HalfVector([1.5, 2, 3]), HalfVector([4.5, 5, 6])]
        cur.execute('INSERT INTO psycopg2_items (half_embeddings) VALUES (%s::halfvec[])', (embeddings,))

        cur.execute('SELECT half_embeddings FROM psycopg2_items ORDER BY id')
        res = cur.fetchone()
        assert res[0][0].to_list() == [1.5, 2, 3]
        assert res[0][1].to_list() == [4.5, 5, 6]

    def test_sparsevec_array(self):
        embeddings = [SparseVector([1.5, 2, 3]), SparseVector([4.5, 5, 6])]
        cur.execute('INSERT INTO psycopg2_items (sparse_embeddings) VALUES (%s::sparsevec[])', (embeddings,))

        cur.execute('SELECT sparse_embeddings FROM psycopg2_items ORDER BY id')
        res = cur.fetchone()
        assert res[0][0].to_list() == [1.5, 2, 3]
        assert res[0][1].to_list() == [4.5, 5, 6]

    def test_cursor_factory(self):
        for cursor_factory in [DictCursor, RealDictCursor, NamedTupleCursor]:
            conn = psycopg2.connect(dbname='pgvector_python_test')
            cur = conn.cursor(cursor_factory=cursor_factory)
            register_vector(cur)
            conn.close()

    def test_cursor_factory_connection(self):
        for cursor_factory in [DictCursor, RealDictCursor, NamedTupleCursor]:
            conn = psycopg2.connect(dbname='pgvector_python_test', cursor_factory=cursor_factory)
            register_vector(conn)
            conn.close()

    def test_pool(self):
        pool = ThreadedConnectionPool(1, 1, dbname='pgvector_python_test')

        conn = pool.getconn()
        try:
            # use globally=True for apps to ensure registered with all connections
            register_vector(conn)
        finally:
            pool.putconn(conn)

        conn = pool.getconn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT '[1,2,3]'::vector")
            res = cur.fetchone()
            assert res[0] == Vector([1, 2, 3])
        finally:
            pool.putconn(conn)

        pool.closeall()
