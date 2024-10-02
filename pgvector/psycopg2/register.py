import psycopg2
from psycopg2.extensions import cursor
from .halfvec import register_halfvec_info
from .sparsevec import register_sparsevec_info
from .vector import register_vector_info


# TODO make globally False by default in 0.4.0
def register_vector(conn_or_curs=None, globally=True):
    conn = conn_or_curs if hasattr(conn_or_curs, 'cursor') else conn_or_curs.connection
    cur = conn.cursor(cursor_factory=cursor)
    scope = None if globally else conn_or_curs

    # use to_regtype to get first matching type in search path
    cur.execute("SELECT typname, oid FROM pg_type WHERE oid IN (to_regtype('vector'), to_regtype('halfvec'), to_regtype('sparsevec'))")
    type_info = dict(cur.fetchall())

    if 'vector' not in type_info:
        raise psycopg2.ProgrammingError('vector type not found in the database')

    register_vector_info(type_info['vector'], scope)

    if 'halfvec' in type_info:
        register_halfvec_info(type_info['halfvec'], scope)

    if 'sparsevec' in type_info:
        register_sparsevec_info(type_info['sparsevec'], scope)
