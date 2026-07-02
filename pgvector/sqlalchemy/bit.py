from sqlalchemy.dialects.postgresql.base import PGBit
from sqlalchemy.types import TypeDecorator, Float
from sqlalchemy import Operators
from typing import Any


class BIT(TypeDecorator[Any]):
    impl = PGBit
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if dialect.__class__.__name__ == 'PGDialect_asyncpg' and isinstance(value, str):
            import asyncpg
            return asyncpg.BitString(value)  # type: ignore
        return value

    class Comparator(TypeDecorator.Comparator[Any]):
        def hamming_distance(self, other: object) -> Operators:
            return self.op('<~>', return_type=Float)(other)

        def jaccard_distance(self, other: object) -> Operators:
            return self.op('<%>', return_type=Float)(other)

    comparator_factory = Comparator  # type: ignore
