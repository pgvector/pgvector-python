# good resources
# https://qdrant.tech/articles/hybrid-search/
# https://www.sbert.net/examples/applications/semantic-search/README.html

import asyncio
import itertools
from pgvector.psycopg import register_vector_async
import psycopg
from sentence_transformers import CrossEncoder, SentenceTransformer
import json

sentences = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]
query = 'growling bear'


async def create_schema(conn):
    await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    await register_vector_async(conn)

    await conn.execute('DROP TABLE IF EXISTS documents')
    await conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(384))')
    await conn.execute("CREATE INDEX ON documents USING GIN (to_tsvector('english', content))")


async def insert_data(conn):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    embeddings = model.encode(sentences)

    sql = 'INSERT INTO documents (content, embedding) VALUES ' + ', '.join(['(%s, %s)' for _ in embeddings])
    params = list(itertools.chain(*zip(sentences, embeddings)))
    await conn.execute(sql, params)


async def semantic_search(conn, query):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    embedding = model.encode(query)
    embedding_json = json.dumps(embedding.tolist())
    async with conn.cursor() as cur:
        await cur.execute('SELECT id, content FROM documents ORDER BY embedding <=> %s LIMIT 5', (embedding_json,))
        return await cur.fetchall()


async def keyword_search(conn, query):
    async with conn.cursor() as cur:
        await cur.execute("SELECT id, content FROM documents, plainto_tsquery('english', %s) query WHERE to_tsvector('english', content) @@ query ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC LIMIT 5", (query,))
        return await cur.fetchall()


def rerank(query, results):
    # deduplicate
    results = set(itertools.chain(*results))

    # re-rank
    encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = encoder.predict([(query, item[1]) for item in results])
    return [v for _, v in sorted(zip(scores, results), reverse=True)]


async def main():
    conn = await psycopg.AsyncConnection.connect(dbname='pgvector_example', autocommit=True)
    await create_schema(conn)
    await insert_data(conn)

    # perform queries in parallel
    results = await asyncio.gather(semantic_search(conn, query), keyword_search(conn, query))
    results = rerank(query, results)
    print(results)


asyncio.run(main())
