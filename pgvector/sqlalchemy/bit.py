from sqlalchemy.dialects.postgresql.base import ischema_names
from sqlalchemy.types import UserDefinedType, TypeEngine, Float
from sqlalchemy import Dialect, Operators
from typing import Any


class BIT(UserDefinedType[Any]):
    cache_ok = True

    def __init__(self, length: int | None = None) -> None:
        super().__init__()
        self.length = length

    def get_col_spec(self, **kw: Any) -> str:
        if self.length is None:
            return 'BIT'
        return 'BIT(%d)' % self.length

    def bind_processor(self, dialect: Dialect) -> Any:
        if dialect.__class__.__name__ == 'PGDialect_asyncpg':
            import asyncpg

            def process(value: Any) -> Any:
                if isinstance(value, str):
                    return asyncpg.BitString(value)  # type: ignore
                return value
            return process
        else:
            return super().bind_processor(dialect)

    class Comparator(TypeEngine.Comparator[Any]):
        def hamming_distance(self, other: object) -> Operators:
            return self.op('<~>', return_type=Float)(other)

        def jaccard_distance(self, other: object) -> Operators:
            return self.op('<%>', return_type=Float)(other)

    comparator_factory = Comparator


# for reflection
ischema_names['bit'] = BIT  # type: ignore
