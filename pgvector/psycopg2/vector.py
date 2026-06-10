from psycopg2.extensions import adapt, connection, cursor, new_array_type, new_type, register_adapter, register_type
from typing import TYPE_CHECKING
from .. import Vector

if TYPE_CHECKING:
    import numpy as np


class VectorAdapter:
    def __init__(self, value: Vector | np.ndarray) -> None:
        if not isinstance(value, Vector):
            value = Vector(value)
        self._value = value

    def getquoted(self) -> bytes:
        return adapt(self._value.to_text()).getquoted()


def cast_vector(value: str | None, cur: cursor) -> Vector | None:
    if value is None:
        return None
    return Vector.from_text(value)


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
