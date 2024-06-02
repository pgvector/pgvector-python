import numpy as np
from struct import pack, unpack_from


class Bit:
    def __init__(self, value):
        if isinstance(value, bytes):
            self._value = __class__.from_binary(value)._value
        elif isinstance(value, str):
            self._value = __class__.from_text(value)._value
        else:
            self._value = np.asarray(value, dtype=bool)

    def __str__(self):
        return self.to_text()

    def __repr__(self):
        return f'Bit({self})'

    def dim(self):
        return self._value.shape[0]

    def to_list(self):
        return self._value.tolist()

    def to_numpy(self):
        return self._value

    def to_text(self):
        return ''.join(self._value.astype(np.uint8).astype(str))

    def to_binary(self):
        value = self._value
        return pack('>i', len(value)) + np.packbits(value).tobytes()

    def from_text(value):
        return Bit(np.asarray([v != '0' for v in value], dtype=bool))

    def from_binary(value):
        count = unpack_from('>i', value)[0]
        buf = np.frombuffer(value[4:], dtype=np.uint8)
        return Bit(np.unpackbits(buf, count=count).astype(bool))

    # TODO move rest

    def to_db(value):
        if not isinstance(value, Bit):
            raise ValueError('expected bit')

        return value.to_text()

    def to_db_binary(value):
        if not isinstance(value, Bit):
            raise ValueError('expected bit')

        return value.to_binary()
