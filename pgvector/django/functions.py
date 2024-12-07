from django.db.models import FloatField, Func, Value
from ..utils import Vector, HalfVector, SparseVector


class DistanceBase(Func):
    output_field = FloatField()

    def __init__(self, expression, vector, **extra):
        if not hasattr(vector, 'resolve_expression'):
            if isinstance(vector, HalfVector):
                vector = Value(HalfVector._to_db(vector))
            elif isinstance(vector, SparseVector):
                vector = Value(SparseVector._to_db(vector))
            else:
                vector = Value(Vector._to_db(vector))
        super().__init__(expression, vector, **extra)


class BitDistanceBase(Func):
    output_field = FloatField()

    def __init__(self, expression, vector, **extra):
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
    
class CosineSimilarity(DistanceBase):
    function = ''
    arg_joiner = ' <=> '
    
    def as_sql(self, compiler, connection):
        sql, params = super().as_sql(compiler, connection)
        return f"1 - ({sql})", params


class L1Distance(DistanceBase):
    function = ''
    arg_joiner = ' <+> '


class HammingDistance(BitDistanceBase):
    function = ''
    arg_joiner = ' <~> '


class JaccardDistance(BitDistanceBase):
    function = ''
    arg_joiner = ' <%%> '
