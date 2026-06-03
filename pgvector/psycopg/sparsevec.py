from psycopg import BaseConnection
from psycopg.adapt import Loader, Dumper
from psycopg.pq import Format
from psycopg.types import TypeInfo
from typing import Any, TypeAlias
from .. import SparseVector

Buffer: TypeAlias = bytes | bytearray | memoryview


class SparseVectorDumper(Dumper):
    format = Format.TEXT

    def dump(self, obj: SparseVector) -> Buffer | None:
        value = SparseVector._to_db(obj)
        return value if value is None else value.encode('utf8')


class SparseVectorBinaryDumper(SparseVectorDumper):
    format = Format.BINARY

    def dump(self, obj: SparseVector) -> Buffer | None:
        return SparseVector._to_db_binary(obj)


class SparseVectorLoader(Loader):
    format = Format.TEXT

    def load(self, data: Buffer) -> SparseVector | None:
        if isinstance(data, memoryview):
            data = bytes(data)
        return SparseVector._from_db(data.decode('utf8'))


class SparseVectorBinaryLoader(SparseVectorLoader):
    format = Format.BINARY

    def load(self, data: Buffer) -> SparseVector | None:
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        return SparseVector._from_db_binary(data)


def register_sparsevec_info(context: BaseConnection[Any], info: TypeInfo) -> None:
    info.register(context)

    # add oid to anonymous class for set_types
    text_dumper = type('', (SparseVectorDumper,), {'oid': info.oid})
    binary_dumper = type('', (SparseVectorBinaryDumper,), {'oid': info.oid})

    adapters = context.adapters
    adapters.register_dumper(SparseVector, text_dumper)
    adapters.register_dumper(SparseVector, binary_dumper)
    adapters.register_loader(info.oid, SparseVectorLoader)
    adapters.register_loader(info.oid, SparseVectorBinaryLoader)
