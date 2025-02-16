from pgvector.psycopg import register_vector
import psycopg
from sentence_transformers import SentenceTransformer

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(384))')

model = SentenceTransformer('all-MiniLM-L6-v2')

input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]
embeddings = model.encode(input)
for content, embedding in zip(input, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))

query = 'forest'
query_embedding = model.encode(query)
result = conn.execute('SELECT content FROM documents ORDER BY embedding <=> %s LIMIT 5', (query_embedding,)).fetchall()
for row in result:
    print(row[0])
