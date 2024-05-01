import psycopg
from psycopg.types import TypeInfo
from .halfvec import register_halfvec_info
from .sparsevec import register_sparsevec_info
from .vector import register_vector_info
from ..utils import HalfVec, SparseVec

# TODO remove in 0.3.0
from .vector import *
from ..utils import from_db, from_db_binary, to_db, to_db_binary

__all__ = ['register_vector']


def register_vector(context):
    info = TypeInfo.fetch(context, 'vector')
    register_vector_info(context, info)

    info = TypeInfo.fetch(context, 'halfvec')
    if info is not None:
        register_halfvec_info(context, info)

    info = TypeInfo.fetch(context, 'sparsevec')
    if info is not None:
        register_sparsevec_info(context, info)


async def register_vector_async(context):
    info = await TypeInfo.fetch(context, 'vector')
    register_vector_info(context, info)

    info = await TypeInfo.fetch(context, 'halfvec')
    if info is not None:
        register_halfvec_info(context, info)

    info = await TypeInfo.fetch(context, 'sparsevec')
    if info is not None:
        register_sparsevec_info(context, info)
