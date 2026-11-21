from sqlalchemy.dialects.postgresql.base import PGBit
from sqlalchemy.types import TypeDecorator, Float, String
from sqlalchemy import Dialect, Operators
from typing import Any
from .. import Bit


class BIT(TypeDecorator[Any]):
    impl = PGBit
    cache_ok = True
    _string = String()

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if isinstance(value, Bit):
            value = value.to_text()
        if dialect.__class__.__name__ == 'PGDialect_asyncpg' and isinstance(value, str):
            import asyncpg
            return asyncpg.BitString(value)  # type: ignore
        return value

    def literal_processor(self, dialect: Dialect) -> Any:
        string_literal_processor = self._string._cached_literal_processor(dialect)

        def process(value: Any) -> Any:
            if isinstance(value, Bit):
                value = value.to_text()
            return string_literal_processor(value)  # type: ignore
        return process

    class Comparator(TypeDecorator.Comparator[Any]):  # type: ignore
        def hamming_distance(self, other: object, /) -> Operators:
            return self.op('<~>', return_type=Float)(other)

        def jaccard_distance(self, other: object, /) -> Operators:
            return self.op('<%>', return_type=Float)(other)

    comparator_factory = Comparator  # type: ignore
