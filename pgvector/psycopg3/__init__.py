import numpy as np
from psycopg3.adapt import Loader, Dumper
from psycopg3.pq import Format
from ..utils import from_db, from_db_binary, to_db, to_db_binary

__all__ = ['register_vector']


class VectorDumper(Dumper):

    format = Format.TEXT

    def dump(self, obj):
        return to_db(obj).encode("utf8")


class VectorBinaryDumper(VectorDumper):

    format = Format.BINARY

    def dump(self, obj):
        return to_db_binary(obj)


class VectorLoader(Loader):

    format = Format.TEXT

    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return from_db(data.decode("utf8"))


class VectorBinaryLoader(VectorLoader):

    format = Format.BINARY

    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return from_db_binary(data)


def register_vector(ctx):
    cur = ctx.cursor() if hasattr(ctx, 'cursor') else ctx

    try:
        cur.execute('SELECT NULL::vector')
        oid = cur.description[0][1]
    except psycopg3.errors.UndefinedObject:
        raise psycopg3.ProgrammingError('vector type not found in the database')

    VectorDumper.register('numpy.ndarray', ctx)
    VectorBinaryDumper.register('numpy.ndarray', ctx)
    VectorLoader.register(oid, ctx)
    VectorBinaryLoader.register(oid, ctx)
