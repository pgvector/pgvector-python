from sqlalchemy.dialects.postgresql.base import ischema_names
from sqlalchemy.types import UserDefinedType, Float
from sqlalchemy import Dialect
from typing import Any


class BIT(UserDefinedType):
    cache_ok = True

    def __init__(self, length: int | None = None) -> None:
        super(UserDefinedType, self).__init__()
        self.length = length

    def get_col_spec(self, **kw) -> str:
        if self.length is None:
            return 'BIT'
        return 'BIT(%d)' % self.length

    def bind_processor(self, dialect: Dialect) -> Any:
        if dialect.__class__.__name__ == 'PGDialect_asyncpg':
            import asyncpg

            def process(value):
                if isinstance(value, str):
                    return asyncpg.BitString(value)
                return value
            return process
        else:
            return super().bind_processor(dialect)

    class comparator_factory(UserDefinedType.Comparator):
        def hamming_distance(self, other: Any) -> Any:
            return self.op('<~>', return_type=Float)(other)

        def jaccard_distance(self, other: Any) -> Any:
            return self.op('<%>', return_type=Float)(other)


# for reflection
ischema_names['bit'] = BIT  # type: ignore
