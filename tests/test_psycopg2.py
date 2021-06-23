import numpy as np
from pgvector.psycopg2 import register_vector
import psycopg2

conn = psycopg2.connect(dbname='pgvector_python_test')
conn.autocommit = True

cur = conn.cursor()
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
cur.execute('DROP TABLE IF EXISTS item')
cur.execute('CREATE TABLE item (id bigserial primary key, factors vector(3))')

register_vector(cur)


class TestPsycopg2(object):
    def test_works(self):
        factors = np.array([1.5, 2, 3])
        cur.execute('INSERT INTO item (factors) VALUES (%s), (NULL)', (factors,))

        cur.execute('SELECT * FROM item ORDER BY id')
        res = cur.fetchall()
        assert res[0][0] == 1
        assert res[1][0] == 2
        assert np.array_equal(res[0][1], factors)
        assert res[0][1].dtype == np.float32
        assert res[1][1] is None
