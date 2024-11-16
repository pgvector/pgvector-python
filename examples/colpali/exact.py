from colpali_engine.models import ColQwen2, ColQwen2Processor
from datasets import load_dataset
from pgvector.psycopg import register_vector, Bit
import psycopg
import torch

conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, embeddings bit(128)[])')
conn.execute("""
CREATE OR REPLACE FUNCTION max_sim(document bit[], query bit[]) RETURNS double precision AS $$
    WITH queries AS (
        SELECT row_number() OVER () AS query_number, * FROM (SELECT unnest(query) AS query)
    ),
    documents AS (
        SELECT unnest(document) AS document
    ),
    similarities AS (
        SELECT query_number, 1 - ((document <~> query) / bit_length(query)) AS similarity FROM queries CROSS JOIN documents
    ),
    max_similarities AS (
        SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number
    )
    SELECT SUM(max_similarity) FROM max_similarities
$$ LANGUAGE SQL
""")


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = ColQwen2.from_pretrained('vidore/colqwen2-v1.0', torch_dtype=torch.bfloat16, device_map=device).eval()
processor = ColQwen2Processor.from_pretrained('vidore/colqwen2-v1.0')


def generate_embeddings(processed):
    with torch.no_grad():
        return model(**processed.to(model.device)).to(device='cpu', dtype=torch.float32)


def binary_quantize(embedding):
    return Bit(embedding > 0)


input = load_dataset('vidore/docvqa_test_subsampled', split='test[:3]')['image']
for content in input:
    embeddings = [binary_quantize(e.numpy()) for e in generate_embeddings(processor.process_images([content]))[0]]
    conn.execute('INSERT INTO documents (embeddings) VALUES (%s)', (embeddings,))

query = 'dividend'
query_embeddings = [binary_quantize(e.numpy()) for e in generate_embeddings(processor.process_queries([query]))[0]]
result = conn.execute('SELECT id, max_sim(embeddings, %s) AS max_sim FROM documents ORDER BY max_sim DESC LIMIT 5', (query_embeddings,)).fetchall()
for row in result:
    print(row)
