from pgvector.psycopg import register_vector
import psycopg
from sentence_transformers import SentenceTransformer

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(384))')

sentences = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

for content, embedding in zip(sentences, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))

document_id = 1
neighbors = conn.execute('SELECT content FROM documents WHERE id != %(id)s ORDER BY embedding <=> (SELECT embedding FROM documents WHERE id = %(id)s) LIMIT 5', {'id': document_id}).fetchall()
for neighbor in neighbors:
    print(neighbor[0])
