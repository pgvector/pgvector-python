import numpy as np
from struct import pack, unpack_from


class Vector:
    def __init__(self, value):
        # asarray still copies if same dtype
        if not isinstance(value, np.ndarray) or value.dtype != '>f4':
            value = np.asarray(value, dtype='>f4')

        if value.ndim != 1:
            raise ValueError('expected ndim to be 1')

        self._value = value

    def __repr__(self):
        return f'Vector({self.to_list()})'

    def dim(self):
        return self._value.shape[0]

    def to_list(self):
        return self._value.tolist()

    def to_numpy(self):
        return self._value

    def to_text(self):
        return '[' + ','.join([str(v) for v in self._value]) + ']'

    def to_binary(self):
        return pack('>HH', self.dim(), 0) + self._value.tobytes()

    def from_text(value):
        return Vector([float(v) for v in value[1:-1].split(',')])

    def from_binary(value):
        dim, unused = unpack_from('>HH', value)
        return Vector(np.frombuffer(value, dtype='>f4', count=dim, offset=4))

    # TODO move rest

    def to_db(value, dim=None):
        if value is None:
            return value

        if not isinstance(value, Vector):
            value = Vector(value)

        if dim is not None and value.dim() != dim:
            raise ValueError('expected %d dimensions, not %d' % (dim, value.dim()))

        return value.to_text()

    def to_db_binary(value):
        if value is None:
            return value

        if not isinstance(value, Vector):
            value = Vector(value)

        return value.to_binary()

    def from_db(value):
        if value is None or isinstance(value, np.ndarray):
            return value

        return Vector.from_text(value).to_numpy().astype(np.float32)

    def from_db_binary(value):
        if value is None or isinstance(value, np.ndarray):
            return value

        return Vector.from_binary(value).to_numpy().astype(np.float32)
