import argparse
import statistics
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark nearest-neighbor search against pgvector table')
    parser.add_argument('--dsn', required=True, help='Postgres connection string')
    parser.add_argument('--table', default='items', help='Table containing embeddings')
    parser.add_argument('--queries', type=int, default=100, help='Number of queries to run')
    parser.add_argument('--k', type=int, default=10, help='Result count per query')
    parser.add_argument('--distance', choices=['l2', 'cosine', 'ip'], default='cosine', help='Distance metric')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--ivfflat-probes', type=int, default=None, help='Optional ivfflat.probes session value')
    parser.add_argument('--hnsw-ef-search', type=int, default=None, help='Optional hnsw.ef_search session value')
    return parser.parse_args()


def operator(distance: str) -> str:
    return {
        'l2': '<->',
        'cosine': '<=>',
        'ip': '<#>',
    }[distance]


def main() -> None:
    args = parse_args()

    conn = psycopg.connect(args.dsn, autocommit=True)
    register_vector(conn)

    if args.ivfflat_probes is not None:
        conn.execute(f'SET ivfflat.probes = {args.ivfflat_probes}')
    if args.hnsw_ef_search is not None:
        conn.execute(f'SET hnsw.ef_search = {args.hnsw_ef_search}')

    dims = conn.execute(f'SELECT vector_dims(embedding) FROM {args.table} LIMIT 1').fetchone()
    if dims is None:
        raise RuntimeError(f'Table {args.table} has no rows')

    dimensions = dims[0]
    print(f'Running {args.queries} queries against {args.table} (dimensions={dimensions}, k={args.k})')

    rng = np.random.default_rng(args.seed)
    sql = f'SELECT id FROM {args.table} ORDER BY embedding {operator(args.distance)} %s LIMIT {args.k}'

    latencies = []
    for i in range(args.queries):
        query = rng.random(dimensions, dtype=np.float32)
        started = time.perf_counter()
        conn.execute(sql, (query,)).fetchall()
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        print(f'Query {i + 1}/{args.queries}: {elapsed_ms:.2f} ms')

    print('\nLatency summary (ms)')
    print(f'avg:    {statistics.mean(latencies):.2f}')
    print(f'p50:    {statistics.median(latencies):.2f}')
    print(f'p95:    {np.percentile(latencies, 95):.2f}')
    print(f'p99:    {np.percentile(latencies, 99):.2f}')
    print(f'min:    {min(latencies):.2f}')
    print(f'max:    {max(latencies):.2f}')


if __name__ == '__main__':
    main()
