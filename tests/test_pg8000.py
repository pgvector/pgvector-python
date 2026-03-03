import numpy as np
import os
from pgvector import HalfVector, SparseVector, Vector
from pgvector.pg8000 import register_vector
from pg8000.native import Connection

class TestPg8000:
    def setup_method(self):
        user = os.environ.get("USER", "postgres")
        port = int(os.environ.get("PGPORT", 5432))
        host = os.environ.get("PGHOST", "localhost")
        if host.startswith("/"):
            host = "localhost"
        self.conn = Connection(user, host=host, port=port, password="password", database='pgvector_python_test')
        self.conn.run('CREATE EXTENSION IF NOT EXISTS vector')
        self.conn.run('DROP TABLE IF EXISTS pg8000_items')
        self.conn.run('CREATE TABLE pg8000_items (id bigserial PRIMARY KEY, embedding vector(3), half_embedding halfvec(3), binary_embedding bit(3), sparse_embedding sparsevec(3))')
        register_vector(self.conn)
        self.conn.run('DELETE FROM pg8000_items')

    def teardown_method(self):
        self.conn.close()

    def test_vector(self):
        embedding = np.array([1.5, 2, 3])
        self.conn.run('INSERT INTO pg8000_items (embedding) VALUES (:embedding), (NULL)', embedding=embedding)

        res = self.conn.run('SELECT embedding FROM pg8000_items ORDER BY id')
        assert np.array_equal(res[0][0], embedding)
        assert res[0][0].dtype == np.float32
        assert res[1][0] is None

    def test_vector_class(self):
        embedding = Vector([1.5, 2, 3])
        self.conn.run('INSERT INTO pg8000_items (embedding) VALUES (:embedding), (NULL)', embedding=embedding)

        res = self.conn.run('SELECT embedding FROM pg8000_items ORDER BY id')
        assert np.array_equal(res[0][0], embedding.to_numpy())
        assert res[0][0].dtype == np.float32
        assert res[1][0] is None

    def test_halfvec(self):
        embedding = HalfVector([1.5, 2, 3])
        self.conn.run('INSERT INTO pg8000_items (half_embedding) VALUES (:embedding), (NULL)', embedding=embedding)

        res = self.conn.run('SELECT half_embedding FROM pg8000_items ORDER BY id')
        assert res[0][0] == embedding
        assert res[1][0] is None

    def test_bit(self):
        embedding = '101'
        self.conn.run('INSERT INTO pg8000_items (binary_embedding) VALUES (:embedding), (NULL)', embedding=embedding)

        res = self.conn.run('SELECT binary_embedding FROM pg8000_items ORDER BY id')
        assert res[0][0] == '101'
        assert res[1][0] is None

    def test_sparsevec(self):
        embedding = SparseVector([1.5, 2, 3])
        self.conn.run('INSERT INTO pg8000_items (sparse_embedding) VALUES (:embedding), (NULL)', embedding=embedding)

        res = self.conn.run('SELECT sparse_embedding FROM pg8000_items ORDER BY id')
        assert res[0][0] == embedding
        assert res[1][0] is None
