from __future__ import annotations
import psycopg
from psycopg import BaseConnection
from psycopg.abc import Buffer
from psycopg.adapt import Loader, Dumper
from psycopg.pq import Format
from psycopg.types import TypeInfo
from typing import Any
from .. import Vector
from .._utils import ndarray


class VectorDumper(Dumper):
    format = Format.TEXT

    def dump(self, obj: Vector | ndarray) -> Buffer | None:
        if not isinstance(obj, Vector):
            obj = Vector(obj)
        return obj.to_text().encode('utf8')


class VectorBinaryDumper(VectorDumper):
    format = Format.BINARY

    def dump(self, obj: Vector | ndarray) -> Buffer | None:
        if not isinstance(obj, Vector):
            obj = Vector(obj)
        return obj.to_binary()


class VectorLoader(Loader):
    format = Format.TEXT

    def load(self, data: Buffer) -> Vector | None:
        if isinstance(data, memoryview):
            data = data.tobytes()
        return Vector.from_text(data.decode('utf8'))


class VectorBinaryLoader(VectorLoader):
    format = Format.BINARY

    def load(self, data: Buffer) -> Vector | None:
        return Vector.from_binary(data)


def register_vector_info(context: BaseConnection[Any], info: TypeInfo | None, /) -> None:
    if info is None:
        raise psycopg.ProgrammingError('vector type not found in the database')
    info.register(context)

    # add oid to anonymous class for set_types
    text_dumper = type('', (VectorDumper,), {'oid': info.oid})
    binary_dumper = type('', (VectorBinaryDumper,), {'oid': info.oid})

    adapters = context.adapters
    adapters.register_dumper(Vector, text_dumper)
    adapters.register_dumper(Vector, binary_dumper)
    adapters.register_dumper('numpy.ndarray', text_dumper)
    adapters.register_dumper('numpy.ndarray', binary_dumper)
    adapters.register_loader(info.oid, VectorLoader)
    adapters.register_loader(info.oid, VectorBinaryLoader)
