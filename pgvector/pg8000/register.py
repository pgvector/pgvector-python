from pg8000.native import Connection
from typing import cast
from .. import Vector, HalfVector, SparseVector


def register_vector(conn: Connection, /) -> None:
    # use to_regtype to get first matching type in search path
    res = cast(list[tuple[str, int]], conn.run("SELECT typname, oid FROM pg_type WHERE oid IN (to_regtype('vector'), to_regtype('halfvec'), to_regtype('sparsevec'))"))
    type_info = dict(res)

    if 'vector' not in type_info:
        raise RuntimeError('vector type not found in the database')

    conn.register_out_adapter(Vector, lambda v: v.to_text())
    conn.register_in_adapter(type_info['vector'], Vector.from_text)

    try:
        import numpy as np
        conn.register_out_adapter(np.ndarray, lambda v: Vector(v).to_text())
    except ImportError:
        pass

    if 'halfvec' in type_info:
        conn.register_out_adapter(HalfVector, lambda v: v.to_text())
        conn.register_in_adapter(type_info['halfvec'], HalfVector.from_text)

    if 'sparsevec' in type_info:
        conn.register_out_adapter(SparseVector, lambda v: v.to_text())
        conn.register_in_adapter(type_info['sparsevec'], SparseVector.from_text)
