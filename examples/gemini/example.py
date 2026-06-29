from google import genai
from pgvector import Vector
from pgvector.psycopg import register_vector
import psycopg

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
# text-embedding-004 has 768 dimensions by default
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(768))')


def embed(input_texts):
    client = genai.Client()
    response = client.models.embed_content(
        model='text-embedding-004',
        contents=input_texts
    )
    return [e.values for e in response.embeddings]


input_data = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]
embeddings = embed(input_data)
for content, embedding in zip(input_data, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, Vector(embedding)))

query = 'forest'
query_embedding = embed([query])[0]
result = conn.execute('SELECT content FROM documents ORDER BY embedding <=> %s LIMIT 5', (Vector(query_embedding),)).fetchall()
for row in result:
    print(row[0])
