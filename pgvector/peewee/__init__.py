from peewee import Expression, Field, Node, Value

from pgvector.utils import from_db, to_db

L2_DIST = ' <-> '
MAX_INNER_PROD = ' <#> '
COS_DIST = ' <=> '


class VectorValue(Node):
    def __init__(self, field, value):
        self.field = field
        self.value = value

    def __sql__(self, ctx):
        return (
            ctx.sql(Value(self.value, unpack=False))
            .literal('::' + self.field.field_type)
        )


class VectorField(Field):
    field_type = 'VECTOR'

    def __init__(self, *args, dimensions=None, **kwargs):
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    db_value = staticmethod(to_db)
    python_value = staticmethod(from_db)

    def _distance(self, vector, op):
        return Expression(lhs=self, op=op, rhs=VectorValue(self, vector))

    def l2_distance(self, vector):
        return self._distance(vector, op=L2_DIST)

    def cosine_distance(self, vector):
        return self._distance(vector, op=COS_DIST)

    def max_inner_product(self, vector):
        return self._distance(vector, op=MAX_INNER_PROD)
