import numpy as np
from struct import pack, unpack_from


class Bit:
    def __init__(self, value):
        if isinstance(value, str):
            self._value = self.from_text(value)._value
        else:
            if isinstance(value, np.ndarray):
                if value.dtype == np.uint8:
                    value = np.unpackbits(value).astype(bool)
                elif value.dtype != np.bool:
                    raise ValueError('expected dtype to be bool or uint8')
            else:
                value = np.asarray(value, dtype=bool)

            if value.ndim != 1:
                raise ValueError('expected ndim to be 1')

            self._value = value

    def __repr__(self):
        return f'Bit({self.to_text()})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.array_equal(self.to_numpy(), other.to_numpy())
        return False

    def to_list(self):
        return self._value.tolist()

    def to_numpy(self):
        return self._value

    def to_text(self):
        return ''.join(self._value.astype(np.uint8).astype(str))

    def to_binary(self):
        return pack('>i', len(self._value)) + np.packbits(self._value).tobytes()

    @classmethod
    def from_text(cls, value):
        return cls(np.asarray([v != '0' for v in value], dtype=bool))

    @classmethod
    def from_binary(cls, value):
        count = unpack_from('>i', value)[0]
        buf = np.frombuffer(value, dtype=np.uint8, offset=4)
        return cls(np.unpackbits(buf, count=count).astype(bool))

    @classmethod
    def _to_db(cls, value):
        if not isinstance(value, cls):
            raise ValueError('expected bit')

        return value.to_text()

    @classmethod
    def _to_db_binary(cls, value):
        if not isinstance(value, cls):
            raise ValueError('expected bit')

        return value.to_binary()
