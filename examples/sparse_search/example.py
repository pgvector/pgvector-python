# good resources
# https://opensearch.org/blog/improving-document-retrieval-with-sparse-semantic-encoders/
# https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v1

import numpy as np
from pgvector.psycopg import register_vector, SparseVector
import psycopg
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding sparsevec(30522))')

model_id = 'opensearch-project/opensearch-neural-sparse-encoding-v1'
model = AutoModelForMaskedLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
special_token_ids = [tokenizer.vocab[token] for token in tokenizer.special_tokens_map.values()]


def fetch_embeddings(input):
    feature = tokenizer(
        input,
        padding=True,
        truncation=True,
        return_tensors='pt',
        return_token_type_ids=False
    )
    output = model(**feature)[0]

    values, _ = torch.max(output * feature['attention_mask'].unsqueeze(-1), dim=1)
    values = torch.log(1 + torch.relu(values))
    values[:, special_token_ids] = 0
    return values.detach().cpu().numpy()


# note: works much better with longer content
input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]
embeddings = fetch_embeddings(input)
for content, embedding in zip(input, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, SparseVector(embedding)))

query = 'forest'
query_embedding = fetch_embeddings([query])[0]
result = conn.execute('SELECT content FROM documents ORDER BY embedding <#> %s LIMIT 5', (SparseVector(query_embedding),)).fetchall()
for row in result:
    print(row[0])
