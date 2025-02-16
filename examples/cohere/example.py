import cohere
import numpy as np
from pgvector.psycopg import register_vector, Bit
import psycopg

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding bit(1024))')


def embed(input, input_type):
    co = cohere.Client()
    response = co.embed(texts=input, model='embed-english-v3.0', input_type=input_type, embedding_types=['ubinary'])
    return [np.unpackbits(np.array(embedding, dtype=np.uint8)) for embedding in response.embeddings.ubinary]


input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]
embeddings = embed(input, 'search_document')
for content, embedding in zip(input, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, Bit(embedding)))

query = 'forest'
query_embedding = embed([query], 'search_query')[0]
result = conn.execute('SELECT content FROM documents ORDER BY embedding <~> %s LIMIT 5', (Bit(query_embedding),)).fetchall()
for row in result:
    print(row[0])
