from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from pgvector.psycopg import register_vector
import psycopg

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embeddings vector(128)[])')
conn.execute("""
CREATE OR REPLACE FUNCTION max_sim(document vector[], query vector[]) RETURNS double precision AS $$
    WITH queries AS (
        SELECT row_number() OVER () AS query_number, * FROM (SELECT unnest(query) AS query)
    ),
    documents AS (
        SELECT unnest(document) AS document
    ),
    similarities AS (
        SELECT query_number, 1 - (document <=> query) AS similarity FROM queries CROSS JOIN documents
    ),
    max_similarities AS (
        SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number
    )
    SELECT SUM(max_similarity) FROM max_similarities
$$ LANGUAGE SQL
""")

checkpoint = Checkpoint('colbert-ir/colbertv2.0', colbert_config=ColBERTConfig(), verbose=0)

input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]
doc_embeddings = checkpoint.docFromText(input)
for content, embeddings in zip(input, doc_embeddings):
    embeddings = [e.numpy() for e in embeddings if e.count_nonzero() > 0]
    conn.execute('INSERT INTO documents (content, embeddings) VALUES (%s, %s)', (content, embeddings))

query = 'puppy'
query_embeddings = [e.numpy() for e in checkpoint.queryFromText([query])[0]]
result = conn.execute('SELECT content, max_sim(embeddings, %s) AS max_sim FROM documents ORDER BY max_sim DESC LIMIT 5', (query_embeddings,)).fetchall()
for row in result:
    print(row)
