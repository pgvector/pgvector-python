import numpy as np
from pgvector.psycopg3 import register_vector
import psycopg3

conn = psycopg3.connect(dbname='pgvector_python_test')
conn.autocommit = True

cur = conn.cursor()
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
cur.execute('DROP TABLE IF EXISTS item')
cur.execute('CREATE TABLE item (id bigserial primary key, factors vector(3))')

register_vector(cur)


class TestPsycopg3(object):
    def setup_method(self, test_method):
        cur.execute('DELETE FROM item')

    def test_works(self):
        factors = np.array([1.5, 2, 3])
        cur.execute('INSERT INTO item (factors) VALUES (%s), (NULL)', (factors,))

        cur.execute('SELECT * FROM item ORDER BY id')
        res = cur.fetchall()
        assert np.array_equal(res[0][1], factors)
        assert res[0][1].dtype == np.float32
        assert res[1][1] is None

    def test_binary_format(self):
        factors = np.array([1.5, 2, 3])
        res = cur.execute('SELECT %b::vector', (factors,)).fetchone()[0]
        assert np.array_equal(res, factors)

    def test_text_format(self):
        factors = np.array([1.5, 2, 3])
        res = cur.execute('SELECT %t::vector', (factors,)).fetchone()[0]
        assert np.array_equal(res, factors)

    def test_binary_format_correct(self):
        factors = np.array([1.5, 2, 3])
        res = cur.execute('SELECT %b::vector::text', (factors,)).fetchone()[0]
        assert res == '[1.5,2,3]'

    def test_non_contiguous(self):
        factors = np.flipud(np.array([1.5, 2, 3]))
        assert not factors.data.contiguous
        res = cur.execute('SELECT %b::vector', (factors,)).fetchone()[0]
        assert np.array_equal(res, np.array([3, 2, 1.5]))
