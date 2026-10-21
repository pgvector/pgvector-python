from functools import cache
from psycopg import BaseConnection
from psycopg.abc import Buffer
from psycopg.adapt import Dumper
from psycopg.pq import Format
from psycopg.types import TypeInfo
from typing import Any
from .. import Bit


class BitDumper(Dumper):
    format = Format.TEXT

    def dump(self, obj: Bit) -> Buffer | None:
        return obj.to_text().encode('utf8')


class BitBinaryDumper(BitDumper):
    format = Format.BINARY

    def dump(self, obj: Bit) -> Buffer | None:
        return obj.to_binary()


def register_bit_info(context: BaseConnection[Any], info: TypeInfo | None, /) -> None:
    assert info is not None
    info.register(context)

    # add oid to anonymous class for set_types
    text_dumper = bit_text_dumper(info.oid)
    binary_dumper = bit_binary_dumper(info.oid)

    adapters = context.adapters
    adapters.register_dumper(Bit, text_dumper)
    adapters.register_dumper(Bit, binary_dumper)


@cache
def bit_text_dumper(oid: int):
    return type('', (BitDumper,), {'oid': oid})


@cache
def bit_binary_dumper(oid: int):
    return type('', (BitBinaryDumper,), {'oid': oid})
