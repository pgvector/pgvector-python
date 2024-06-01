import numpy as np
from struct import pack, unpack_from


class Bit:
    def __init__(self, value):
        if isinstance(value, bytes):
            count = unpack_from('>i', value)[0]
            buf = np.frombuffer(value[4:], dtype=np.uint8)
            self._value = np.unpackbits(buf, count=count).astype(bool)
        elif isinstance(value, str):
            self._value = np.array([v != '0' for v in value], dtype=bool)
        else:
            self._value = np.array(value, dtype=bool)

    def __str__(self):
        return self.to_text()

    def __repr__(self):
        return f'Bit({self})'

    def to_text(self):
        return ''.join(self._value.astype(np.uint8).astype(str))

    def to_binary(self):
        value = self._value
        return pack('>i', len(value)) + np.packbits(value).tobytes()

    def to_db(value):
        if not isinstance(value, Bit):
            raise ValueError('expected bit')

        return value.to_text()

    def to_db_binary(value):
        if not isinstance(value, Bit):
            raise ValueError('expected bit')

        return value.to_binary()
