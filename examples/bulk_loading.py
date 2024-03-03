import numpy as np
from pgvector.psycopg import register_vector
import psycopg

# generate random data
rows = 1000000
dimensions = 128
embeddings = np.random.rand(rows, dimensions)

# enable extension
conn = psycopg.connect(dbname='pgvector_example', autocommit=True)
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

# create table
conn.execute('DROP TABLE IF EXISTS items')
conn.execute(f'CREATE TABLE items (id bigserial, embedding vector({dimensions}))')

# load data
print(f'Loading {len(embeddings)} rows')
cur = conn.cursor()
for batch in np.array_split(embeddings, len(embeddings) // 10000):
    # show progress
    print('.', end='', flush=True)

    with cur.copy('COPY items (embedding) FROM STDIN WITH (FORMAT BINARY)') as copy:
        # use set_types for binary copy
        # https://www.psycopg.org/psycopg3/docs/basic/copy.html#binary-copy
        copy.set_types(['vector'])

        for embedding in batch:
            copy.write_row([embedding])

print('\nSuccess!')

# create any indexes *after* loading initial data (skipping for this example)
# print('Creating index')
# conn.execute("SET maintenance_work_mem = '8GB'")
# conn.execute("SET max_parallel_maintenance_workers = 7")
# conn.execute('CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)')

# update planner statistics for good measure
conn.execute('ANALYZE items')
