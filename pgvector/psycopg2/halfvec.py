from psycopg2.extensions import adapt, connection, cursor, new_array_type, new_type, register_adapter, register_type
from typing import Any
from .. import HalfVector


class HalfvecAdapter:
    def __init__(self, value: Any) -> None:
        self._value = value

    def getquoted(self) -> Any:
        return adapt(HalfVector._to_db(self._value)).getquoted()


def cast_halfvec(value: str | None, cur: cursor) -> HalfVector | None:
    return HalfVector._from_db(value)


def register_halfvec_info(oid: int, array_oid: int | None, scope: connection | cursor | None) -> None:
    halfvec = new_type((oid,), 'HALFVEC', cast_halfvec)
    register_type(halfvec, scope)

    if array_oid is not None:
        halfvecarray = new_array_type((array_oid,), 'HALFVECARRAY', halfvec)
        register_type(halfvecarray, scope)

    register_adapter(HalfVector, HalfvecAdapter)
