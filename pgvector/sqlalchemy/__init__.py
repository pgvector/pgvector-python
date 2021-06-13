from sqlalchemy.types import UserDefinedType
from ..utils import from_db, to_db

__all__ = ['Vector']


class Vector(UserDefinedType):
    def __init__(self, dim=None):
        super(UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kw):
        if self.dim is None:
            return "VECTOR"
        return "VECTOR(%d)" % self.dim

    def bind_processor(self, dialect):
        def process(value):
            return to_db(value, self.dim)
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            return from_db(value)
        return process

    class comparator_factory(UserDefinedType.Comparator):
        def l2_distance(self, other):
            return self.op('<->')(other)

        def max_inner_product(self, other):
            return self.op('<#>')(other)

        def cosine_distance(self, other):
            return self.op('<=>')(other)
