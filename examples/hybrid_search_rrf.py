from pgvector.psycopg import register_vector
import psycopg
from sentence_transformers import SentenceTransformer

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(384))')
conn.execute("CREATE INDEX ON documents USING GIN (to_tsvector('english', content))")

sentences = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
embeddings = model.encode(sentences)
for content, embedding in zip(sentences, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))

sql = """
WITH semantic_search AS (
    SELECT id, RANK () OVER (ORDER BY embedding <=> %(embedding)s) AS rank
    FROM documents
    ORDER BY embedding <=> %(embedding)s
    LIMIT 20
),
keyword_search AS (
    SELECT id, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC)
    FROM documents, plainto_tsquery('english', %(query)s) query
    WHERE to_tsvector('english', content) @@ query
    ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC
    LIMIT 20
)
SELECT
    COALESCE(semantic_search.id, keyword_search.id) AS id,
    COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) + COALESCE(1.0 / (%(k)s + keyword_search.rank), 0.0) AS score
FROM semantic_search
FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
ORDER BY score DESC
LIMIT 5
"""
query = 'growling bear'
embedding = model.encode(query)
k = 60
results = conn.execute(sql, {'query': query, 'embedding': embedding, 'k': k}).fetchall()
print(results)
