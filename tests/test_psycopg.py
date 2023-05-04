import numpy as np
from pgvector.psycopg import register_vector
import psycopg

conn = psycopg.connect(dbname='pgvector_python_test')
conn.autocommit = True

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
conn.execute('DROP TABLE IF EXISTS item')
conn.execute('CREATE TABLE item (id bigserial primary key, embedding vector(3))')

register_vector(conn)


class TestPsycopg:
    def setup_method(self, test_method):
        conn.execute('DELETE FROM item')

    def test_works(self):
        embedding = np.array([1.5, 2, 3])
        conn.execute('INSERT INTO item (embedding) VALUES (%s), (NULL)', (embedding,))

        res = conn.execute('SELECT * FROM item ORDER BY id').fetchall()
        assert np.array_equal(res[0][1], embedding)
        assert res[0][1].dtype == np.float32
        assert res[1][1] is None

    def test_binary_format(self):
        embedding = np.array([1.5, 2, 3])
        res = conn.execute('SELECT %b::vector', (embedding,)).fetchone()[0]
        assert np.array_equal(res, embedding)

    def test_text_format(self):
        embedding = np.array([1.5, 2, 3])
        res = conn.execute('SELECT %t::vector', (embedding,)).fetchone()[0]
        assert np.array_equal(res, embedding)

    def test_binary_format_correct(self):
        embedding = np.array([1.5, 2, 3])
        res = conn.execute('SELECT %b::vector::text', (embedding,)).fetchone()[0]
        assert res == '[1.5,2,3]'

    def test_text_format_non_contiguous(self):
        embedding = np.flipud(np.array([1.5, 2, 3]))
        assert not embedding.data.contiguous
        res = conn.execute('SELECT %t::vector', (embedding,)).fetchone()[0]
        assert np.array_equal(res, np.array([3, 2, 1.5]))

    def test_binary_format_non_contiguous(self):
        embedding = np.flipud(np.array([1.5, 2, 3]))
        assert not embedding.data.contiguous
        res = conn.execute('SELECT %b::vector', (embedding,)).fetchone()[0]
        assert np.array_equal(res, np.array([3, 2, 1.5]))

    def test_text_copy(self):
        embedding = np.array([1.5, 2, 3])
        cur = conn.cursor()
        with cur.copy("COPY item (embedding) FROM STDIN") as copy:
            copy.write_row([embedding])

    def test_binary_copy(self):
        embedding = np.array([1.5, 2, 3])
        cur = conn.cursor()
        with cur.copy("COPY item (embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.write_row([embedding])

    def test_binary_copy_set_types(self):
        embedding = np.array([1.5, 2, 3])
        cur = conn.cursor()
        with cur.copy("COPY item (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(['int8', 'vector'])
            copy.write_row([1, embedding])
