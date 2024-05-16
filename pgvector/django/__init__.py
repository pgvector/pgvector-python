from django.contrib.postgres.operations import CreateExtension
from django.db.models import Field, FloatField, Func, Value
import numpy as np
from .indexes import IvfflatIndex, HnswIndex
from .vector import VectorField
from ..utils import Vector

__all__ = ['VectorExtension', 'VectorField', 'IvfflatIndex', 'HnswIndex', 'L2Distance', 'MaxInnerProduct', 'CosineDistance', 'L1Distance']


class VectorExtension(CreateExtension):
    def __init__(self):
        self.name = 'vector'


class DistanceBase(Func):
    output_field = FloatField()

    def __init__(self, expression, vector, **extra):
        if not hasattr(vector, 'resolve_expression'):
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
