# Benchmark scripts

These scripts help benchmark loading and search for large datasets with `pgvector`.

## Pass disk path directly

You can now pass filesystem paths directly to the loader script.
The script auto-creates (or reuses) PostgreSQL tablespaces behind the scenes.

- `--table-path /mnt/nvme/pg_tblspc_fast_table` for table data
- `--index-path /mnt/nvme/pg_tblspc_fast_index` for index data

> Note: creating tablespaces requires sufficient PostgreSQL privileges (typically superuser).

If you prefer, you can still pass existing tablespace names:

- `--table-tablespace fast_nvme`
- `--index-tablespace fast_nvme`

## 1) Load random embeddings

```sh
python examples/benchmark/load_embeddings.py \
  --dsn "postgresql://localhost/pgvector_example" \
  --table benchmark_items \
  --rows 24000000 \
  --dimensions 4096 \
  --batch-size 5000 \
  --drop-table \
  --table-path /mnt/nvme/pg_tblspc_fast_table \
  --index hnsw \
  --index-path /mnt/nvme/pg_tblspc_fast_index \
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

## Progress bars

Both scripts support tqdm progress bars.

Install:

```sh
pip install tqdm
```

If `tqdm` is not installed, scripts still run with a minimal fallback.

Notes:

- Run indexing only after loading data for better performance.
- For IVFFlat, tune `--ivfflat-lists` during index creation and `--ivfflat-probes` during search.
- 24M vectors x 4096 dimensions is very large; ensure enough disk and memory for PostgreSQL and indexes.
