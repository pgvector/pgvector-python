from psycopg2.extensions import adapt, connection, cursor, new_array_type, new_type, register_adapter, register_type
from .. import Vector


class VectorAdapter:
    def __init__(self, value: object) -> None:
        self._value = value

    def getquoted(self) -> bytes:
        return adapt(Vector._to_db(self._value)).getquoted()


def cast_vector(value: str | None, cur: cursor) -> Vector | None:
    return Vector._from_db(value)


def register_vector_info(oid: int, array_oid: int | None, scope: connection | cursor | None) -> None:
    vector = new_type((oid,), 'VECTOR', cast_vector)
    register_type(vector, scope)

    if array_oid is not None:
        vectorarray = new_array_type((array_oid,), 'VECTORARRAY', vector)
        register_type(vectorarray, scope)

    register_adapter(Vector, VectorAdapter)

    try:
        import numpy as np
        register_adapter(np.ndarray, VectorAdapter)
    except ImportError:
        pass
