import numpy as np
from pgvector.psycopg2 import register_vector
import psycopg2

conn = psycopg2.connect(dbname='pgvector_python_test')
conn.autocommit = True

cur = conn.cursor()
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
cur.execute('DROP TABLE IF EXISTS items')
cur.execute('CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3))')

register_vector(cur)


class TestPsycopg2:
    def setup_method(self, test_method):
        cur.execute('DELETE FROM items')

    def test_works(self):
        embedding = np.array([1.5, 2, 3])
        cur.execute('INSERT INTO items (embedding) VALUES (%s), (NULL)', (embedding,))

        cur.execute('SELECT * FROM items ORDER BY id')
        res = cur.fetchall()
        assert np.array_equal(res[0][1], embedding)
        assert res[0][1].dtype == np.float32
        assert res[1][1] is None
