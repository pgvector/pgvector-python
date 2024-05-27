import numpy as np
from struct import pack, unpack_from


class Bit:
    def __init__(self, value):
        if isinstance(value, bytes):
            count = unpack_from('>i', value)[0]
            buf = np.frombuffer(value[4:], dtype=np.uint8)
            self.value = np.unpackbits(buf, count=count).astype(bool)
        elif isinstance(value, str):
            self.value = np.array([v != '0' for v in value], dtype=bool)
        else:
            self.value = np.array(value, dtype=bool)

    def __str__(self):
        return self.__class__.to_db(self)

    def __repr__(self):
        return f'Bit({self})'

    def to_db(value):
        if not isinstance(value, Bit):
            raise ValueError('expected bit')

        value = value.value
        return ''.join(value.astype(np.uint8).astype(str))

    def to_db_binary(value):
        if not isinstance(value, Bit):
            raise ValueError('expected bit')

        value = value.value
        return pack('>i', len(value)) + np.packbits(value).tobytes()
