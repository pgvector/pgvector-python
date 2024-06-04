import numpy as np
from struct import pack, unpack_from


import numpy as np
from struct import pack, unpack_from


class HalfVector:
    def __init__(self, value):
        # asarray still copies if same dtype
        if not isinstance(value, np.ndarray) or value.dtype != '>f2':
            value = np.asarray(value, dtype='>f2')

        if value.ndim != 1:
            raise ValueError('Expected ndim to be 1')

        self._value = value

    def __repr__(self):
        return f'HalfVector({self.to_list()})'

    def dim(self):
        return len(self._value)

    def to_list(self):
        return self._value.tolist()

    def to_numpy(self):
        return self._value

    def to_text(self):
        return '[' + ','.join([str(float(v)) for v in self._value]) + ']'

    def to_binary(self):
        return pack('>HH', self.dim(), 0) + self._value.tobytes()

    @classmethod
    def from_text(cls, value):
        return cls([float(v) for v in value[1:-1].split(',')])

    @classmethod
    def from_binary(cls, value):
        dim, unused = unpack_from('>HH', value)
        return cls(np.frombuffer(value, dtype='>f2', count=dim, offset=4))

    @staticmethod
    def _to_db(value, dim=None):
        if value is None:
            return value

        if not isinstance(value, HalfVector):
            value = HalfVector(value)

        if dim is not None and value.dim() != dim:
            raise ValueError(f'Expected {dim} dimensions, not {value.dim()}')

        return value.to_text()

    @staticmethod
    def _to_db_binary(value):
        if value is None:
            return value

        if not isinstance(value, HalfVector):
            value = HalfVector(value)

        return value.to_binary()

    @classmethod
    def _from_db(cls, value):
        if value is None or isinstance(value, HalfVector):
            return value

        return cls.from_text(value)

    @classmethod
    def _from_db_binary(cls, value):
        if value is None or isinstance(value, HalfVector):
            return value

        return cls.from_binary(value)