from psycopg import AsyncConnection, Connection
from psycopg.types import TypeInfo
from typing import Any
from .bit import register_bit_info
from .halfvec import register_halfvec_info
from .sparsevec import register_sparsevec_info
from .vector import register_vector_info


def register_vector(context: Connection[Any]) -> None:
    info = TypeInfo.fetch(context, 'vector')
    register_vector_info(context, info)

    info = TypeInfo.fetch(context, 'bit')
    assert info is not None
    register_bit_info(context, info)

    info = TypeInfo.fetch(context, 'halfvec')
    if info is not None:
        register_halfvec_info(context, info)

    info = TypeInfo.fetch(context, 'sparsevec')
    if info is not None:
        register_sparsevec_info(context, info)


async def register_vector_async(context: AsyncConnection[Any]) -> None:
    info = await TypeInfo.fetch(context, 'vector')
    register_vector_info(context, info)

    info = await TypeInfo.fetch(context, 'bit')
    assert info is not None
    register_bit_info(context, info)

    info = await TypeInfo.fetch(context, 'halfvec')
    if info is not None:
        register_halfvec_info(context, info)

    info = await TypeInfo.fetch(context, 'sparsevec')
    if info is not None:
        register_sparsevec_info(context, info)
