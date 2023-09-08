import openai
from pgvector.psycopg import register_vector
import psycopg
from sentence_transformers import SentenceTransformer

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(1536))')

input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]

response = openai.Embedding.create(input=input, model='text-embedding-ada-002')
embeddings = [v['embedding'] for v in response['data']]

for content, embedding in zip(input, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))

document_id = 1
neighbors = conn.execute('SELECT content FROM documents WHERE id != %(id)s ORDER BY embedding <=> (SELECT embedding FROM documents WHERE id = %(id)s) LIMIT 5', {'id': document_id}).fetchall()
for neighbor in neighbors:
    print(neighbor[0])
