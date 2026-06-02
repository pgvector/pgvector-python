from peewee import Expression, Field
from typing import Any
from .. import SparseVector


class SparseVectorField(Field):
    field_type = 'sparsevec'

    def __init__(self, dimensions: int | None = None, *args: Any, **kwargs: Any) -> None:
        self.dimensions = dimensions
        super(SparseVectorField, self).__init__(*args, **kwargs)

    def get_modifiers(self) -> list[int] | None:
        return [self.dimensions] if self.dimensions else None

    def db_value(self, value: object) -> str | None:
        return SparseVector._to_db(value)

    def python_value(self, value: Any) -> SparseVector | None:
        return SparseVector._from_db(value)

    def _distance(self, op: str, vector: object) -> Expression:
        return Expression(lhs=self, op=op, rhs=self.to_value(vector))

    def l2_distance(self, vector: object) -> Expression:
        return self._distance('<->', vector)

    def max_inner_product(self, vector: object) -> Expression:
        return self._distance('<#>', vector)

    def cosine_distance(self, vector: object) -> Expression:
        return self._distance('<=>', vector)

    def l1_distance(self, vector: object) -> Expression:
        return self._distance('<+>', vector)
