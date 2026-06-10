from psycopg import BaseConnection
from psycopg.types import TypeInfo
from psycopg.adapt import Dumper
from psycopg.pq import Format
from typing import Any, TypeAlias
from .. import Bit

Buffer: TypeAlias = bytes | bytearray | memoryview


class BitDumper(Dumper):
    format = Format.TEXT

    def dump(self, obj: Bit) -> Buffer | None:
        return obj.to_text().encode('utf8')


class BitBinaryDumper(BitDumper):
    format = Format.BINARY

    def dump(self, obj: Bit) -> Buffer | None:
        return obj.to_binary()


def register_bit_info(context: BaseConnection[Any], info: TypeInfo | None) -> None:
    assert info is not None
    info.register(context)

    # add oid to anonymous class for set_types
    text_dumper = type('', (BitDumper,), {'oid': info.oid})
    binary_dumper = type('', (BitBinaryDumper,), {'oid': info.oid})

    adapters = context.adapters
    adapters.register_dumper(Bit, text_dumper)
    adapters.register_dumper(Bit, binary_dumper)
