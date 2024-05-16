import psycopg2
from .vector import register_vector_info
from ..utils import from_db, to_db

__all__ = ['register_vector']


def register_vector(conn_or_curs=None):
    cur = conn_or_curs.cursor() if hasattr(conn_or_curs, 'cursor') else conn_or_curs

    try:
        cur.execute('SELECT NULL::vector')
        oid = cur.description[0][1]
    except psycopg2.errors.UndefinedObject:
        raise psycopg2.ProgrammingError('vector type not found in the database')

    register_vector_info(oid)
