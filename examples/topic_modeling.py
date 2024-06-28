from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
import numpy as np
from pgvector.psycopg import register_vector
import psycopg

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(20))')

input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]

docs = [simple_preprocess(content) for content in input]
dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=1)
corpus = [dictionary.doc2bow(tokens) for tokens in docs]
model = LdaModel(corpus, num_topics=20)

for content, bow in zip(input, corpus):
    embedding = np.array([v[1] for v in model.get_document_topics(bow, minimum_probability=0)])
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))

document_id = 1
neighbors = conn.execute('SELECT content FROM documents WHERE id != %(id)s ORDER BY embedding <=> (SELECT embedding FROM documents WHERE id = %(id)s) LIMIT 5', {'id': document_id}).fetchall()
for neighbor in neighbors:
    print(neighbor[0])
