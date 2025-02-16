import numpy as np
from openai import OpenAI
from pgvector.psycopg import register_vector
import psycopg

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(1536))')


def embed(input):
    client = OpenAI()
    response = client.embeddings.create(input=input, model='text-embedding-3-small')
    return [v.embedding for v in response.data]


input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]
embeddings = embed(input)
for content, embedding in zip(input, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, np.array(embedding)))

query = 'forest'
query_embedding = embed([query])[0]
result = conn.execute('SELECT content FROM documents ORDER BY embedding <=> %s LIMIT 5', (np.array(query_embedding),)).fetchall()
for row in result:
    print(row[0])
