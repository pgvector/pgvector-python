from getpass import getuser
from pgvector import HalfVector, SparseVector, Vector
from pgvector.pg8000 import register_vector
from pg8000.native import Connection
import pytest

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

conn = Connection(getuser(), database='pgvector_python_test')

conn.run('CREATE EXTENSION IF NOT EXISTS vector')
conn.run('DROP TABLE IF EXISTS pg8000_items')
conn.run('CREATE TABLE pg8000_items (id bigserial PRIMARY KEY, embedding vector(3), half_embedding halfvec(3), binary_embedding bit(3), sparse_embedding sparsevec(3))')

register_vector(conn)


class TestPg8000:
    def setup_method(self):
        conn.run('DELETE FROM pg8000_items')

    def test_vector(self):
        embedding = Vector([1.5, 2, 3])
        conn.run('INSERT INTO pg8000_items (embedding) VALUES (:embedding), (NULL)', embedding=embedding)

        res = conn.run('SELECT embedding FROM pg8000_items ORDER BY id')
        assert res[0][0] == embedding
        assert res[1][0] is None

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason='NumPy required')
    def test_vector_numpy(self):
        embedding = np.array([1.5, 2, 3])
        conn.run('INSERT INTO pg8000_items (embedding) VALUES (:embedding), (NULL)', embedding=embedding)

        res = conn.run('SELECT embedding FROM pg8000_items ORDER BY id')
        assert res[0][0] == Vector([1.5, 2, 3])
        assert res[1][0] is None

    def test_halfvec(self):
        embedding = HalfVector([1.5, 2, 3])
        conn.run('INSERT INTO pg8000_items (half_embedding) VALUES (:embedding), (NULL)', embedding=embedding)

        res = conn.run('SELECT half_embedding FROM pg8000_items ORDER BY id')
        assert res[0][0] == embedding
        assert res[1][0] is None

    def test_bit(self):
        embedding = '101'
        conn.run('INSERT INTO pg8000_items (binary_embedding) VALUES (:embedding), (NULL)', embedding=embedding)

        res = conn.run('SELECT binary_embedding FROM pg8000_items ORDER BY id')
        assert res[0][0] == '101'
        assert res[1][0] is None

    def test_sparsevec(self):
        embedding = SparseVector([1.5, 2, 3])
        conn.run('INSERT INTO pg8000_items (sparse_embedding) VALUES (:embedding), (NULL)', embedding=embedding)

        res = conn.run('SELECT sparse_embedding FROM pg8000_items ORDER BY id')
        assert res[0][0] == embedding
        assert res[1][0] is None
