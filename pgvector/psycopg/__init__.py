import psycopg
from psycopg.types import TypeInfo
from .vector import *

# TODO remove in 0.3.0
from ..utils import from_db, from_db_binary, to_db, to_db_binary

__all__ = ['register_vector']


def register_vector(context):
    info = TypeInfo.fetch(context, 'vector')
    register_vector_info(context, info)


async def register_vector_async(context):
    info = await TypeInfo.fetch(context, 'vector')
    register_vector_info(context, info)
