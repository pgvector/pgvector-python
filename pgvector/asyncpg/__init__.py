from ..utils import from_db, from_db_binary, to_db, to_db_binary, HalfVec, SparseVec

__all__ = ['register_vector']


async def register_vector(conn):
    await conn.set_type_codec(
        'vector',
        encoder=to_db_binary,
        decoder=from_db_binary,
        format='binary'
    )

    await conn.set_type_codec(
        'halfvec',
        encoder=HalfVec.to_db_binary,
        decoder=HalfVec.from_db_binary,
        format='binary'
    )

    await conn.set_type_codec(
        'sparsevec',
        encoder=SparseVec.to_db_binary,
        decoder=SparseVec.from_db_binary,
        format='binary'
    )
