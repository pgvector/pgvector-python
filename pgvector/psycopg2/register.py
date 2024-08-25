import psycopg2
from .halfvec import register_halfvec_info
from .sparsevec import register_sparsevec_info
from .vector import register_vector_info


def get_type_info(cur):
    # use to_regtype to get first matching type in search path
    cur.execute("SELECT typname, oid FROM pg_type WHERE oid IN (to_regtype('vector'), to_regtype('halfvec'), to_regtype('sparsevec'))")
    results = cur.fetchall()

    try:
        return {r[0]: r[1] for r in results}
    except KeyError:
        return {r['typname']: r['oid'] for r in results}


def register_vector(conn_or_curs=None):
    cur = conn_or_curs.cursor() if hasattr(conn_or_curs, 'cursor') else conn_or_curs

    type_info = get_type_info(cur)

    if 'vector' not in type_info:
        raise psycopg2.ProgrammingError('vector type not found in the database')

    register_vector_info(type_info['vector'])

    if 'halfvec' in type_info:
        register_halfvec_info(type_info['halfvec'])

    if 'sparsevec' in type_info:
        register_sparsevec_info(type_info['sparsevec'])
