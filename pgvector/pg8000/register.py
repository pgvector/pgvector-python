from pg8000.native import Connection
from .. import Vector, HalfVector, SparseVector


def register_vector(conn: Connection) -> None:
    # use to_regtype to get first matching type in search path
    res = conn.run("SELECT typname, oid FROM pg_type WHERE oid IN (to_regtype('vector'), to_regtype('halfvec'), to_regtype('sparsevec'))")
    type_info = dict(res)

    if 'vector' not in type_info:
        raise RuntimeError('vector type not found in the database')

    conn.register_out_adapter(Vector, Vector._to_db)
    conn.register_in_adapter(type_info['vector'], Vector.from_text)

    try:
        import numpy as np
        conn.register_out_adapter(np.ndarray, Vector._to_db)
    except ImportError:
        pass

    if 'halfvec' in type_info:
        conn.register_out_adapter(HalfVector, HalfVector._to_db)
        conn.register_in_adapter(type_info['halfvec'], HalfVector.from_text)

    if 'sparsevec' in type_info:
        conn.register_out_adapter(SparseVector, SparseVector._to_db)
        conn.register_in_adapter(type_info['sparsevec'], SparseVector.from_text)
