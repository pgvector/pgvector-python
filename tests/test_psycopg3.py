import numpy as np
from pgvector.psycopg3 import register_vector
import psycopg3


class TestPsycopg3(object):
    def test_works(self):
        conn = psycopg3.connect('dbname=pgvector_python_test')
        conn.autocommit = True

        cur = conn.cursor()
        cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        cur.execute('DROP TABLE IF EXISTS item')
        cur.execute('CREATE TABLE item (id bigserial primary key, factors vector(3))')

        register_vector(cur)

        factors = np.array([1.5, 2, 3])
        cur.execute("INSERT INTO item (factors) VALUES (%s), (NULL)", (factors,))

        cur.execute("SELECT * FROM item ORDER BY id")
        res = cur.fetchall()
        assert res[0][0] == 1
        assert res[1][0] == 2
        assert np.array_equal(res[0][1], factors)
        assert res[0][1].dtype == np.float32
        assert res[1][1] is None

        # binary format
        binary_res = cur.execute("SELECT %b::vector", (factors,)).fetchone()[0]
        assert np.array_equal(binary_res, factors)

        # text format
        text_res = cur.execute("SELECT %t::vector", (factors,)).fetchone()[0]
        assert np.array_equal(text_res, factors)

        # ensures binary format is correct
        binary_text_res = cur.execute("SELECT %b::vector::text", (factors,)).fetchone()[0]
        assert binary_text_res == '[1.5,2,3]'
