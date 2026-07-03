from asyncpg import Connection
from .. import Vector, HalfVector, SparseVector


async def register_vector(conn: Connection, /, *, schema: str = 'public') -> None:
    await conn.set_type_codec(
        'vector',
        schema=schema,
        encoder=lambda v: (v if isinstance(v, Vector) else Vector(v)).to_binary(),
        decoder=Vector.from_binary,
        format='binary'
    )

    try:
        await conn.set_type_codec(
            'halfvec',
            schema=schema,
            encoder=lambda v: (v if isinstance(v, HalfVector) else HalfVector(v)).to_binary(),
            decoder=HalfVector.from_binary,
            format='binary'
        )

        await conn.set_type_codec(
            'sparsevec',
            schema=schema,
            encoder=lambda v: (v if isinstance(v, SparseVector) else SparseVector(v)).to_binary(),
            decoder=SparseVector.from_binary,
            format='binary'
        )
    except ValueError as e:
        if not str(e).startswith('unknown type:'):
            raise e
