import psycopg2
from .bit import register_bit_info
from .halfvec import register_halfvec_info
from .sparsevec import register_sparsevec_info
from .vector import register_vector_info
from ..utils import Bit, SparseVector

__all__ = ['register_vector']


def register_vector(conn_or_curs=None):
    cur = conn_or_curs.cursor() if hasattr(conn_or_curs, 'cursor') else conn_or_curs

    try:
        cur.execute('SELECT NULL::vector')
        register_vector_info(cur.description[0][1])
    except psycopg2.errors.UndefinedObject:
        raise psycopg2.ProgrammingError('vector type not found in the database')

    register_bit_info()

    try:
        cur.execute('SELECT NULL::halfvec')
        register_halfvec_info(cur.description[0][1])
    except psycopg2.errors.UndefinedObject:
        pass

    try:
        cur.execute('SELECT NULL::sparsevec')
        register_sparsevec_info(cur.description[0][1])
    except psycopg2.errors.UndefinedObject:
        pass
