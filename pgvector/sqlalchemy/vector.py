from sqlalchemy.dialects.postgresql.base import ischema_names
from sqlalchemy.types import UserDefinedType, Float, String
from sqlalchemy import Dialect, Operators
from typing import Any
from .. import Vector


class VECTOR(UserDefinedType):
    cache_ok = True
    _string = String()

    def __init__(self, dim: int | None = None) -> None:
        super(UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kw: Any) -> str:
        if self.dim is None:
            return 'VECTOR'
        return 'VECTOR(%d)' % self.dim

    def bind_processor(self, dialect: Dialect) -> Any:
        def process(value: Any) -> str | None:
            return Vector._to_db(value, self.dim)
        return process

    def literal_processor(self, dialect: Dialect) -> Any:
        string_literal_processor = self._string._cached_literal_processor(dialect)

        def process(value: Any) -> Any:
            return string_literal_processor(Vector._to_db(value, self.dim))  # type: ignore
        return process

    def result_processor(self, dialect: Dialect, coltype: Any) -> Any:
        def process(value: Any) -> Vector | None:
            return Vector._from_db(value)
        return process

    class comparator_factory(UserDefinedType.Comparator):
        def l2_distance(self, other: object) -> Operators:
            return self.op('<->', return_type=Float)(other)

        def max_inner_product(self, other: object) -> Operators:
            return self.op('<#>', return_type=Float)(other)

        def cosine_distance(self, other: object) -> Operators:
            return self.op('<=>', return_type=Float)(other)

        def l1_distance(self, other: object) -> Operators:
            return self.op('<+>', return_type=Float)(other)


# for reflection
ischema_names['vector'] = VECTOR  # type: ignore
