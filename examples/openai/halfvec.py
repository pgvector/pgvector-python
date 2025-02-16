from openai import OpenAI
from pgvector.psycopg import register_vector, HalfVector
import psycopg

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding halfvec(3072))')
conn.execute('CREATE INDEX ON documents USING hnsw (embedding halfvec_cosine_ops)')


def embed(input):
    client = OpenAI()
    response = client.embeddings.create(input=input, model='text-embedding-3-large')
    return [v.embedding for v in response.data]


input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]
embeddings = embed(input)
for content, embedding in zip(input, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, HalfVector(embedding)))

query = 'forest'
query_embedding = embed([query])[0]
result = conn.execute('SELECT content FROM documents ORDER BY embedding <=> %s LIMIT 5', (HalfVector(query_embedding),)).fetchall()
for row in result:
    print(row[0])
