from ..utils import Vector, HalfVector, SparseVector

__all__ = ['register_vector']


async def register_vector(conn):
    await conn.set_type_codec(
        'vector',
        encoder=Vector.to_db_binary,
        decoder=Vector.from_db_binary,
        format='binary'
    )

    await conn.set_type_codec(
        'halfvec',
        encoder=HalfVector.to_db_binary,
        decoder=HalfVector.from_db_binary,
        format='binary'
    )

    await conn.set_type_codec(
        'sparsevec',
        encoder=SparseVector.to_db_binary,
        decoder=SparseVector.from_db_binary,
        format='binary'
    )
