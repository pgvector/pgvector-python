from psycopg2.extensions import adapt, connection, cursor, new_array_type, new_type, register_adapter, register_type
from .. import SparseVector


class SparsevecAdapter:
    def __init__(self, value: object) -> None:
        self._value = value

    def getquoted(self) -> bytes:
        return adapt(SparseVector._to_db(self._value)).getquoted()


def cast_sparsevec(value: str | None, cur: cursor) -> SparseVector | None:
    return SparseVector._from_db(value)


def register_sparsevec_info(oid: int, array_oid: int | None, scope: connection | cursor | None) -> None:
    sparsevec = new_type((oid,), 'SPARSEVEC', cast_sparsevec)
    register_type(sparsevec, scope)

    if array_oid is not None:
        sparsevecarray = new_array_type((array_oid,), 'SPARSEVECARRAY', sparsevec)
        register_type(sparsevecarray, scope)

    register_adapter(SparseVector, SparsevecAdapter)
