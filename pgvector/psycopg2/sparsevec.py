from psycopg2.extensions import adapt, connection, cursor, new_array_type, new_type, register_adapter, register_type
from typing import Any
from .. import SparseVector


class SparsevecAdapter:
    def __init__(self, value: SparseVector) -> None:
        self._value = value

    def getquoted(self) -> Any:
        return adapt(self._value.to_text()).getquoted()


def cast_sparsevec(value: str | bytes | None, cur: cursor) -> SparseVector | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        raise ValueError('expected str')
    return SparseVector.from_text(value)


def register_sparsevec_info(oid: int, array_oid: int | None, scope: connection | cursor | None, /) -> None:
    sparsevec = new_type((oid,), 'SPARSEVEC', cast_sparsevec)
    register_type(sparsevec, scope)

    if array_oid is not None:
        sparsevecarray = new_array_type((array_oid,), 'SPARSEVECARRAY', sparsevec)
        register_type(sparsevecarray, scope)

    register_adapter(SparseVector, SparsevecAdapter)
