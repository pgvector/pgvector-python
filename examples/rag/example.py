# Run:
# ollama pull llama3.2
# ollama pull nomic-embed-text
# ollama serve

import numpy as np
import ollama
from pathlib import Path
from pgvector.psycopg import register_vector
import psycopg
import urllib.request

query = 'What index types are supported?'
load_data = True

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

if load_data:
    # get data
    url = 'https://raw.githubusercontent.com/pgvector/pgvector/refs/heads/master/README.md'
    dest = Path(__file__).parent / 'README.md'
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)

    with open(dest, encoding='utf-8') as f:
        doc = f.read()

    # generate chunks
    # TODO improve chunking
    # TODO remove markdown
    chunks = doc.split('\n## ')

    # embed chunks
    # nomic-embed-text has task instruction prefix
    input = ['search_document: ' + chunk for chunk in chunks]
    embeddings = ollama.embed(model='nomic-embed-text', input=input).embeddings

    # create table
    conn.execute('DROP TABLE IF EXISTS chunks')
    conn.execute('CREATE TABLE chunks (id bigserial PRIMARY KEY, content text, embedding vector(768))')

    # store chunks
    cur = conn.cursor()
    with cur.copy('COPY chunks (content, embedding) FROM STDIN WITH (FORMAT BINARY)') as copy:
        copy.set_types(['text', 'vector'])

        for content, embedding in zip(chunks, embeddings):
            copy.write_row([content, embedding])

# embed query
# nomic-embed-text has task instruction prefix
input = 'search_query: ' + query
embedding = ollama.embed(model='nomic-embed-text', input=input).embeddings[0]

# retrieve chunks
result = conn.execute('SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 5', (np.array(embedding),)).fetchall()
context = '\n\n'.join([row[0] for row in result])

# get answer
# TODO improve prompt
prompt = f'Answer this question: {query}\n\n{context}'
response = ollama.generate(model='llama3.2', prompt=prompt).response
print(response)
