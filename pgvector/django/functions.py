from django.db.models import FloatField, Func, Value
from ..utils import Vector, HalfVec, SparseVec


class DistanceBase(Func):
    output_field = FloatField()

    def __init__(self, expression, vector, **extra):
        if not hasattr(vector, 'resolve_expression'):
            if isinstance(vector, HalfVec):
                vector = Value(HalfVec.to_db(vector))
            elif isinstance(vector, SparseVec):
                vector = Value(SparseVec.to_db(vector))
            else:
                vector = Value(Vector.to_db(vector))
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
