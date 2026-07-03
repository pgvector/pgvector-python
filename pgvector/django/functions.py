from django.db.models import FloatField, Func, Value
from .. import Vector, HalfVector, SparseVector
from typing import Any


class DistanceBase(Func):
    output_field = FloatField()  # type: ignore

    def __init__(self, expression: Any, vector: Any, /, **extra: Any) -> None:
        if not hasattr(vector, 'resolve_expression'):
            if isinstance(vector, (Vector, HalfVector, SparseVector)):
                vector = Value(vector.to_text())
            elif vector is not None:
                vector = Value(Vector(vector).to_text())

            # prevent error with unhashable types
            self._constructor_args = ((expression, vector), extra)

        super().__init__(expression, vector, **extra)


class BitDistanceBase(Func):
    output_field = FloatField()  # type: ignore

    def __init__(self, expression: Any, vector: Any, /, **extra: Any) -> None:
        if not hasattr(vector, 'resolve_expression'):
            vector = Value(vector)
        super().__init__(expression, vector, **extra)


class L2Distance(DistanceBase):
    function = ''
    arg_joiner = ' <-> '


class MaxInnerProduct(DistanceBase):
    function = ''
    arg_joiner = ' <#> '


class CosineDistance(DistanceBase):
    function = ''
    arg_joiner = ' <=> '


class L1Distance(DistanceBase):
    function = ''
    arg_joiner = ' <+> '


class HammingDistance(BitDistanceBase):
    function = ''
    arg_joiner = ' <~> '


class JaccardDistance(BitDistanceBase):
    function = ''
    arg_joiner = ' <%%> '
