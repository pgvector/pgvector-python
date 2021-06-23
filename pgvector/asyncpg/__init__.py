from ..utils import from_db, to_db


async def register_vector(conn):
    await conn.set_type_codec(
        'vector',
        encoder=to_db,
        decoder=from_db,
        format='text'
    )
