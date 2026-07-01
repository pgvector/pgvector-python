from getpass import getuser
from pgvector import HalfVector, SparseVector, Vector
from pgvector.pg8000 import register_vector
from pg8000.native import Connection

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
    def setup_method(self) -> None:
        conn.run('DELETE FROM pg8000_items')

    def test_vector(self) -> None:
        embedding = Vector([1.5, 2, 3])
        embedding2 = np.array([4.5, 5, 6]) if NUMPY_AVAILABLE else Vector([4.5, 5, 6])
        embedding3 = None
        conn.run('INSERT INTO pg8000_items (embedding) VALUES (:embedding), (:embedding2), (:embedding3)', embedding=embedding, embedding2=embedding2, embedding3=embedding3)

        res = conn.run('SELECT embedding FROM pg8000_items ORDER BY id')
        assert res == [[embedding], [Vector([4.5, 5, 6])], [None]]

    def test_halfvec(self) -> None:
        embedding = HalfVector([1.5, 2, 3])
        embedding2 = None
        conn.run('INSERT INTO pg8000_items (half_embedding) VALUES (:embedding), (:embedding2)', embedding=embedding, embedding2=embedding2)

        res = conn.run('SELECT half_embedding FROM pg8000_items ORDER BY id')
        assert res == [[embedding], [None]]

    def test_bit(self) -> None:
        embedding = '101'
        embedding2 = None
        conn.run('INSERT INTO pg8000_items (binary_embedding) VALUES (:embedding), (:embedding2)', embedding=embedding, embedding2=embedding2)

        res = conn.run('SELECT binary_embedding FROM pg8000_items ORDER BY id')
        assert res == [['101'], [None]]

    def test_sparsevec(self) -> None:
        embedding = SparseVector([1.5, 2, 3])
        embedding2 = None
        conn.run('INSERT INTO pg8000_items (sparse_embedding) VALUES (:embedding), (:embedding2)', embedding=embedding, embedding2=embedding2)

        res = conn.run('SELECT sparse_embedding FROM pg8000_items ORDER BY id')
        assert res == [[embedding], [None]]
