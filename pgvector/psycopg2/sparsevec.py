from psycopg2.extensions import adapt, new_type, register_adapter, register_type
from ..utils import SparseVec


class SparsevecAdapter(object):
    def __init__(self, value):
        self._value = value

    def getquoted(self):
        return adapt(SparseVec.to_db(self._value)).getquoted()


def cast_sparsevec(value, cur):
    return SparseVec.from_db(value)


def register_sparsevec_info(oid):
    sparsevec = new_type((oid,), 'SPARSEVEC', cast_sparsevec)
    register_type(sparsevec)
    register_adapter(SparseVec, SparsevecAdapter)
