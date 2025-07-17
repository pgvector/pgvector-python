import asyncpg
from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
from sqlalchemy.dialects.postgresql.base import ischema_names
from sqlalchemy.types import UserDefinedType, Float
from .. import Bit

class BIT(UserDefinedType):
    cache_ok = True

    def __init__(self, length=None):
        super(UserDefinedType, self).__init__()
        self.length = length

    def get_col_spec(self, **kw):
        if self.length is None:
            return 'BIT'
        return 'BIT(%d)' % self.length

    def bind_processor(self, dialect):
        def process(value):
            value = Bit._to_db(value)
            if value and isinstance(dialect, PGDialect_asyncpg): 
                return asyncpg.BitString(value)
            return value
        return process
    
    def result_processor(self, dialect, coltype):
        def process(value): 
            if value is None: return None
            else: 
                if isinstance(dialect, PGDialect_asyncpg): 
                    return value.as_string()
            return Bit._from_db(value).to_text() 
        return process

    class comparator_factory(UserDefinedType.Comparator):
        def hamming_distance(self, other):
            return self.op('<~>', return_type=Float)(other)

        def jaccard_distance(self, other):
            return self.op('<%>', return_type=Float)(other)


# for reflection
ischema_names['bit'] = BIT
