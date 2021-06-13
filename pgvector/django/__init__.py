from django.contrib.postgres.operations import CreateExtension
from django.contrib.postgres.indexes import PostgresIndex
from django.db.models import Field, FloatField, Func
from ..utils import from_db, to_db

__all__ = ['VectorExtension', 'VectorField', 'IvfflatIndex', 'L2Distance', 'MaxInnerProduct', 'CosineDistance']


class VectorExtension(CreateExtension):
    def __init__(self):
        self.name = 'vector'


# https://docs.djangoproject.com/en/3.2/howto/custom-model-fields/
class VectorField(Field):
    description = 'Vector'

    def __init__(self, *args, dimensions=None, **kwargs):
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.dimensions is not None:
            kwargs['dimensions'] = self.dimensions
        return name, path, args, kwargs

    def db_type(self, connection):
        if self.dimensions is None:
            return 'vector'
        return 'vector(%d)' % self.dimensions

    def from_db_value(self, value, expression, connection):
        return from_db(value)

    def to_python(self, value):
        return from_db(value)

    def get_prep_value(self, value):
        return to_db(value)


class IvfflatIndex(PostgresIndex):
    suffix = 'ivfflat'

    def __init__(self, *expressions, lists=None, **kwargs):
        self.lists = lists
        super().__init__(*expressions, **kwargs)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        if self.lists is not None:
            kwargs['lists'] = self.lists
        return path, args, kwargs

    def get_with_params(self):
        with_params = []
        if self.lists is not None:
            with_params.append('lists = %d' % self.lists)
        return with_params


class DistanceBase(Func):
    output_field = FloatField()


class L2Distance(DistanceBase):
    function = ''
    arg_joiner = ' <-> '


class MaxInnerProduct(DistanceBase):
    function = ''
    arg_joiner = ' <#> '


class CosineDistance(DistanceBase):
    function = ''
    arg_joiner = ' <=> '
