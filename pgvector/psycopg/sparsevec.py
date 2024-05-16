from psycopg.adapt import Loader, Dumper
from psycopg.pq import Format
from ..utils import SparseVec


class SparseVecDumper(Dumper):

    format = Format.TEXT

    def dump(self, obj):
        return SparseVec.to_db(obj).encode('utf8')


class SparseVecBinaryDumper(SparseVecDumper):

    format = Format.BINARY

    def dump(self, obj):
        return SparseVec.to_db_binary(obj)


class SparseVecLoader(Loader):

    format = Format.TEXT

    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return SparseVec.from_db(data.decode('utf8'))


class SparseVecBinaryLoader(SparseVecLoader):

    format = Format.BINARY

    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return SparseVec.from_db_binary(data)


def register_sparsevec_info(context, info):
    info.register(context)

    # add oid to anonymous class for set_types
    text_dumper = type('', (SparseVecDumper,), {'oid': info.oid})
    binary_dumper = type('', (SparseVecBinaryDumper,), {'oid': info.oid})

    adapters = context.adapters
    adapters.register_dumper(SparseVec, text_dumper)
    adapters.register_dumper(SparseVec, binary_dumper)
    adapters.register_loader(info.oid, SparseVecLoader)
    adapters.register_loader(info.oid, SparseVecBinaryLoader)
