from ..utils import Vector, HalfVector, SparseVector


async def register_vector(conn):
    await conn.set_type_codec(
        'vector',
        encoder=Vector._to_db_binary,
        decoder=Vector._from_db_binary,
        format='binary'
    )

    try:
        await conn.set_type_codec(
            'halfvec',
            encoder=HalfVector._to_db_binary,
            decoder=HalfVector._from_db_binary,
            format='binary'
        )

        await conn.set_type_codec(
            'sparsevec',
            encoder=SparseVector._to_db_binary,
            decoder=SparseVector._from_db_binary,
            format='binary'
        )
    except ValueError as e:
        if not str(e).startswith('unknown type:'):
            raise e
