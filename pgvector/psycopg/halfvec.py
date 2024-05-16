from psycopg.adapt import Loader, Dumper
from psycopg.pq import Format
from ..utils import HalfVec


class HalfVecDumper(Dumper):

    format = Format.TEXT

    def dump(self, obj):
        return HalfVec.to_db(obj).encode('utf8')


class HalfVecBinaryDumper(HalfVecDumper):

    format = Format.BINARY

    def dump(self, obj):
        return HalfVec.to_db_binary(obj)


class HalfVecLoader(Loader):

    format = Format.TEXT

    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return HalfVec.from_db(data.decode('utf8'))


class HalfVecBinaryLoader(HalfVecLoader):

    format = Format.BINARY

    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return HalfVec.from_db_binary(data)


def register_halfvec_info(context, info):
    info.register(context)

    # add oid to anonymous class for set_types
    text_dumper = type('', (HalfVecDumper,), {'oid': info.oid})
    binary_dumper = type('', (HalfVecBinaryDumper,), {'oid': info.oid})

    adapters = context.adapters
    adapters.register_dumper(HalfVec, text_dumper)
    adapters.register_dumper(HalfVec, binary_dumper)
    adapters.register_loader(info.oid, HalfVecLoader)
    adapters.register_loader(info.oid, HalfVecBinaryLoader)
