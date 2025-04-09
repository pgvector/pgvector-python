# based on section 3.6 of https://arxiv.org/abs/2004.12832

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from pgvector.psycopg import register_vector
import psycopg

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('DROP TABLE IF EXISTS document_embeddings')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text)')
conn.execute('CREATE TABLE document_embeddings (id bigserial PRIMARY KEY, document_id bigint, embedding vector(128))')
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

config = ColBERTConfig(doc_maxlen=220, query_maxlen=32)
checkpoint = Checkpoint('colbert-ir/colbertv2.0', colbert_config=config, verbose=0)

input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]
doc_embeddings = checkpoint.docFromText(input, keep_dims=False)
for content, embeddings in zip(input, doc_embeddings):
    with conn.transaction():
        result = conn.execute('INSERT INTO documents (content) VALUES (%s) RETURNING id', (content,)).fetchone()
        params = []
        for embedding in embeddings:
            params.extend([result[0], embedding.numpy()])
        values = ', '.join(['(%s, %s)' for _ in embeddings])
        conn.execute(f'INSERT INTO document_embeddings (document_id, embedding) VALUES {values}', params)

conn.execute('CREATE INDEX ON document_embeddings (document_id)')
conn.execute('CREATE INDEX ON document_embeddings USING hnsw (embedding vector_cosine_ops)')

query = 'puppy'
query_embeddings = [e.numpy() for e in checkpoint.queryFromText([query])[0]]
approximate_stage = ' UNION ALL '.join(['(SELECT document_id FROM document_embeddings ORDER BY embedding <=> %s LIMIT 5)' for _ in query_embeddings])
sql = f"""
WITH approximate_stage AS (
    {approximate_stage}
),
embeddings AS (
    SELECT document_id, array_agg(embedding) AS embeddings FROM document_embeddings
    WHERE document_id IN (SELECT DISTINCT document_id FROM approximate_stage)
    GROUP BY document_id
)
SELECT content, max_sim(embeddings, %s) AS max_sim FROM documents
INNER JOIN embeddings ON embeddings.document_id = documents.id
ORDER BY max_sim DESC LIMIT 10
"""
params = [v for v in query_embeddings] + [query_embeddings]
result = conn.execute(sql, params).fetchall()
for row in result:
    print(row)
