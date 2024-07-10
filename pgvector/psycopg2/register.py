import psycopg2
from .halfvec import register_halfvec_info
from .sparsevec import register_sparsevec_info
from .vector import register_vector_info


def register_vector(conn_or_curs=None):
    cur = conn_or_curs.cursor() if hasattr(conn_or_curs, 'cursor') else conn_or_curs

    cur.execute("SELECT typname, oid FROM pg_type WHERE typname IN ('vector', 'halfvec', 'sparsevec')")
    type_info = dict(cur.fetchall())

    if 'vector' not in type_info:
        raise psycopg2.ProgrammingError('vector type not found in the database')

    register_vector_info(type_info['vector'])

    if 'halfvec' in type_info:
        register_halfvec_info(type_info['halfvec'])

    if 'sparsevec' in type_info:
        register_sparsevec_info(type_info['sparsevec'])
