from peewee import Expression, Field
from typing import Any


class FixedBitField(Field):
    field_type = 'bit'

    def __init__(self, max_length: int | None = None, *args: Any, **kwargs: Any) -> None:
        self.max_length = max_length
        super(FixedBitField, self).__init__(*args, **kwargs)

    def get_modifiers(self) -> list[int] | None:
        return [self.max_length] if self.max_length else None

    def _distance(self, op: str, vector: object) -> Expression:
        return Expression(lhs=self, op=op, rhs=self.to_value(vector))

    def hamming_distance(self, vector: object) -> Expression:
        return self._distance('<~>', vector)

    def jaccard_distance(self, vector: object) -> Expression:
        return self._distance('<%%>', vector)
