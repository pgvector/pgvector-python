# Benchmark scripts

These scripts help benchmark loading and search for large datasets with `pgvector`.

## 1) Load random embeddings

```sh
python examples/benchmark/load_embeddings.py \
  --dsn "postgresql://localhost/pgvector_example" \
  --table benchmark_items \
  --rows 24000000 \
  --dimensions 4096 \
  --batch-size 5000 \
  --drop-table \
  --index hnsw \
  --distance cosine
```

## 2) Benchmark search latency

```sh
python examples/benchmark/search_embeddings.py \
  --dsn "postgresql://localhost/pgvector_example" \
  --table benchmark_items \
  --queries 100 \
  --k 10 \
  --distance cosine \
  --hnsw-ef-search 100
```

Notes:

- Run indexing only after loading data for better performance.
- For IVFFlat, tune `--ivfflat-lists` during index creation and `--ivfflat-probes` during search.
- 24M vectors x 4096 dimensions is very large; ensure enough disk and memory for PostgreSQL and indexes.
