from __future__ import annotations
from struct import pack, unpack_from

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class Bit:
    def __init__(self, value: bytes | str | list[bool] | np.ndarray[tuple[int], np.dtype[np.bool | np.uint8]]) -> None:
        if isinstance(value, bytes):
            self._length = 8 * len(value)
            self._data = value
        elif isinstance(value, (list, str)):
            if isinstance(value, list):
                def bit_value(v: bool) -> str:
                    if v is True:
                        return '1'
                    if v is False:
                        return '0'
                    raise ValueError('expected list[bool]')

                value = ''.join([bit_value(v) for v in value])

            length = len(value)

            if length % 8 != 0:
                value += '0' * (8 - (length % 8))

            self._length = length
            try:
                self._data = int(value, 2).to_bytes(len(value) // 8, byteorder='big')
            except ValueError:
                raise ValueError('expected bit string')
        elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
            if value.dtype != np.bool:
                # skip error for result of np.unpackbits
                if value.dtype != np.uint8 or np.any(value > 1):
                    raise ValueError('expected elements to be boolean')
                value = value.astype(bool)

            if value.ndim != 1:
                raise ValueError('expected ndim to be 1')

            self._length = len(value)
            self._data = np.packbits(value).tobytes()
        else:
            raise ValueError('expected bytes, str, list, or ndarray')

    def __repr__(self) -> str:
        return f'Bit({self.to_text()})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self._length == other._length and self._data == other._data
        return False

    def to_list(self) -> list[bool]:
        # TODO improve
        return [v != '0' for v in self.to_text()]

    def to_numpy(self) -> np.ndarray[tuple[int], np.dtype[np.bool]]:
        return np.unpackbits(np.frombuffer(self._data, dtype=np.uint8), count=self._length).astype(bool)

    def to_text(self) -> str:
        return ''.join(format(v, '08b') for v in self._data)[:self._length]

    def to_binary(self) -> bytes:
        return pack('>i', self._length) + self._data

    @classmethod
    def from_text(cls, value: str) -> Bit:
        return cls(str(value))

    @classmethod
    def from_binary(cls, value: bytes) -> Bit:
        if not isinstance(value, bytes):
            raise ValueError('expected bytes')

        length, = unpack_from('>i', value)
        data = value[4:]

        if len(data) != (length + 7) // 8:
            raise ValueError('invalid length')

        bit = cls.__new__(cls)
        bit._length = length
        bit._data = data
        return bit
