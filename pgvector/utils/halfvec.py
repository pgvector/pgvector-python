import numpy as np
from struct import pack, unpack_from


class HalfVector:
    def __init__(self, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()

        if not isinstance(value, (list, tuple)):
            raise ValueError('expected list or tuple')

        self._value = value

    def __repr__(self):
        return f'HalfVector({self._value})'

    def to_text(self):
        return '[' + ','.join([str(float(v)) for v in self._value]) + ']'

    def to_binary(self):
        return pack(f'>HH{len(self._value)}e', len(self._value), 0, *self._value)

    def dim(self):
        return len(self._value)

    def to_list(self):
        return list(self._value)

    def from_text(value):
        return HalfVector([float(v) for v in value[1:-1].split(',')])

    def from_binary(value):
        dim, unused = unpack_from('>HH', value)
        return HalfVector(unpack_from(f'>{dim}e', value, 4))

    # TODO move rest

    def to_db(value, dim=None):
        if value is None:
            return value
        if not isinstance(value, HalfVector):
            value = HalfVector(value)

        if dim is not None and value.dim() != dim:
            raise ValueError('expected %d dimensions, not %d' % (dim, value.dim()))

        return value.to_text()

    def to_db_binary(value):
        if value is None:
            return value
        if not isinstance(value, HalfVector):
            value = HalfVector(value)

        return value.to_binary()

    def from_db(value):
        if value is None or isinstance(value, HalfVector):
            return value
        return __class__.from_text(value)

    def from_db_binary(value):
        if value is None or isinstance(value, HalfVector):
            return value
        return __class__.from_binary(value)
