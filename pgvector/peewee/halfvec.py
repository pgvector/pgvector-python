from peewee import Expression, Field
from typing import Any
from .. import HalfVector


class HalfVectorField(Field):
    field_type = 'halfvec'

    def __init__(self, dimensions: int | None = None, *args, **kwargs) -> None:
        self.dimensions = dimensions
        super(HalfVectorField, self).__init__(*args, **kwargs)

    def get_modifiers(self) -> list[int] | None:
        return [self.dimensions] if self.dimensions else None

    def db_value(self, value: Any) -> str | None:
        return HalfVector._to_db(value)

    def python_value(self, value: Any) -> HalfVector | None:
        return HalfVector._from_db(value)

    def _distance(self, op: str, vector: Any) -> Expression:
        return Expression(lhs=self, op=op, rhs=self.to_value(vector))

    def l2_distance(self, vector: Any) -> Expression:
        return self._distance('<->', vector)

    def max_inner_product(self, vector: Any) -> Expression:
        return self._distance('<#>', vector)

    def cosine_distance(self, vector: Any) -> Expression:
        return self._distance('<=>', vector)

    def l1_distance(self, vector: Any) -> Expression:
        return self._distance('<+>', vector)
