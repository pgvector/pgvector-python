import argparse
import hashlib
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    def tqdm(iterable=None, total=None, **kwargs):
        if iterable is None:
            class _SimpleProgress:
                def __init__(self, total=None):
                    self.total = total
                    self.current = 0

                def update(self, n):
                    self.current += n

                def set_postfix_str(self, _):
                    pass

                def close(self):
                    pass

            return _SimpleProgress(total=total)

        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate random vectors and load them into a pgvector table'
    )
    parser.add_argument('--dsn', required=True, help='Postgres connection string')
    parser.add_argument('--table', default='items', help='Destination table name')
    parser.add_argument('--rows', type=int, default=24_000_000, help='Number of rows to generate')
    parser.add_argument('--dimensions', type=int, default=4096, help='Vector dimensions')
    parser.add_argument('--batch-size', type=int, default=5000, help='Rows generated per batch')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--drop-table', action='store_true', help='Drop table before creating it')

    parser.add_argument('--table-tablespace', default=None, help='PostgreSQL tablespace name for table storage')
    parser.add_argument('--index-tablespace', default=None, help='PostgreSQL tablespace name for index storage')

    parser.add_argument(
        '--table-path',
        default=None,
        help='Filesystem path for table storage; script creates/reuses a PostgreSQL tablespace',
    )
    parser.add_argument(
        '--index-path',
        default=None,
        help='Filesystem path for index storage; script creates/reuses a PostgreSQL tablespace',
    )
    parser.add_argument(
        '--tablespace-prefix',
        default='benchmark_ts',
        help='Prefix used when auto-generating tablespace names from --table-path / --index-path',
    )

    parser.add_argument(
        '--index',
        choices=['none', 'hnsw', 'ivfflat'],
        default='none',
        help='Optional index to build after loading',
    )
    parser.add_argument(
        '--distance',
        choices=['l2', 'cosine', 'ip'],
        default='cosine',
        help='Distance metric for index opclass',
    )
    parser.add_argument('--ivfflat-lists', type=int, default=1000, help='lists setting for IVFFlat')
    return parser.parse_args()


def metric_opclass(distance: str) -> str:
    return {
        'l2': 'vector_l2_ops',
        'cosine': 'vector_cosine_ops',
        'ip': 'vector_ip_ops',
    }[distance]


def tablespace_name_for_path(prefix: str, path: str, kind: str) -> str:
    digest = hashlib.sha1(path.encode('utf-8')).hexdigest()[:10]
    return f'{prefix}_{kind}_{digest}'


def ensure_tablespace(conn: psycopg.Connection, name: str, path: str) -> None:
    exists = conn.execute('SELECT 1 FROM pg_tablespace WHERE spcname = %s', (name,)).fetchone()
    if exists:
        print(f'Using existing tablespace {name}')
        return

    print(f'Creating tablespace {name} at {path}')
    conn.execute(
        sql.SQL('CREATE TABLESPACE {} LOCATION {}').format(
            sql.Identifier(name),
            sql.Literal(path),
        )
    )


def resolve_tablespaces(conn: psycopg.Connection, args: argparse.Namespace) -> tuple[str | None, str | None]:
    table_ts = args.table_tablespace
    index_ts = args.index_tablespace

    if args.table_path:
        generated = tablespace_name_for_path(args.tablespace_prefix, args.table_path, 'table')
        if table_ts is not None and table_ts != generated:
            raise ValueError('When using --table-path, do not pass a conflicting --table-tablespace')
        table_ts = generated
        ensure_tablespace(conn, table_ts, args.table_path)

    if args.index_path:
        generated = tablespace_name_for_path(args.tablespace_prefix, args.index_path, 'index')
        if index_ts is not None and index_ts != generated:
            raise ValueError('When using --index-path, do not pass a conflicting --index-tablespace')
        index_ts = generated
        ensure_tablespace(conn, index_ts, args.index_path)

    return table_ts, index_ts


def copy_embeddings(conn: psycopg.Connection, table: str, rows: int, dimensions: int, batch_size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    inserted = 0
    started = time.perf_counter()
    progress = tqdm(total=rows, unit='rows', desc='Loading embeddings')

    with conn.cursor() as cur:
        with cur.copy(f'COPY {table} (embedding) FROM STDIN WITH (FORMAT BINARY)') as copy:
            copy.set_types(['vector'])

            while inserted < rows:
                size = min(batch_size, rows - inserted)
                batch = rng.random((size, dimensions), dtype=np.float32)
                for embedding in batch:
                    copy.write_row([embedding])

                inserted += size
                progress.update(size)
                elapsed = time.perf_counter() - started
                rate = inserted / elapsed if elapsed else 0.0
                progress.set_postfix_str(f'rate={rate:,.0f} rows/s')

    progress.close()


def create_index(
    conn: psycopg.Connection,
    table: str,
    index: str,
    distance: str,
    ivfflat_lists: int,
    index_tablespace: str | None,
) -> None:
    if index == 'none':
        return

    opclass = metric_opclass(distance)

    if index == 'hnsw':
        sql_stmt = f'CREATE INDEX ON {table} USING hnsw (embedding {opclass})'
    else:
        sql_stmt = f'CREATE INDEX ON {table} USING ivfflat (embedding {opclass}) WITH (lists = {ivfflat_lists})'

    if index_tablespace:
        sql_stmt += f' TABLESPACE {index_tablespace}'

    started = time.perf_counter()
    conn.execute(sql_stmt)
    elapsed = time.perf_counter() - started
    print(f'Created {index} index in {elapsed:.2f}s')


def main() -> None:
    args = parse_args()

    conn = psycopg.connect(args.dsn, autocommit=True)
    register_vector(conn)

    conn.execute('CREATE EXTENSION IF NOT EXISTS vector')

    table_tablespace, index_tablespace = resolve_tablespaces(conn, args)

    if args.drop_table:
        conn.execute(f'DROP TABLE IF EXISTS {args.table}')

    create_table_sql = (
        f'CREATE TABLE IF NOT EXISTS {args.table} '
        f'(id bigserial PRIMARY KEY, embedding vector({args.dimensions}))'
    )
    if table_tablespace:
        create_table_sql += f' TABLESPACE {table_tablespace}'
    conn.execute(create_table_sql)

    print(
        f'Loading {args.rows:,} rows with {args.dimensions} dimensions into {args.table} '
        f'using batches of {args.batch_size:,}'
    )

    load_started = time.perf_counter()
    copy_embeddings(conn, args.table, args.rows, args.dimensions, args.batch_size, args.seed)
    load_elapsed = time.perf_counter() - load_started
    print(f'Load finished in {load_elapsed:.2f}s')

    create_index(conn, args.table, args.index, args.distance, args.ivfflat_lists, index_tablespace)

    conn.execute(f'ANALYZE {args.table}')
    print('Done')


if __name__ == '__main__':
    main()
