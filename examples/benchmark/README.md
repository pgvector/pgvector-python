# Benchmark scripts

These scripts help benchmark loading and search for large datasets with `pgvector`.

## About disk location / storage path

You do **not** pass a raw filesystem path directly to the Python scripts.
In PostgreSQL, storage location is controlled with a **tablespace**.

1. Create a tablespace once (as a superuser):

```sql
CREATE TABLESPACE fast_nvme LOCATION '/mnt/nvme/pg_tblspc_fast';
```

2. Pass the tablespace name to the loader script:

- `--table-tablespace fast_nvme` for table data
- `--index-tablespace fast_nvme` for index data

## 1) Load random embeddings

```sh
python examples/benchmark/load_embeddings.py \
  --dsn "postgresql://localhost/pgvector_example" \
  --table benchmark_items \
  --rows 24000000 \
  --dimensions 4096 \
  --batch-size 5000 \
  --drop-table \
  --table-tablespace fast_nvme \
  --index hnsw \
  --index-tablespace fast_nvme \
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
