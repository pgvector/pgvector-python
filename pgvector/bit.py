from __future__ import annotations
import numpy as np
from struct import pack, unpack_from
from warnings import warn


class Bit:
    def __init__(self, value: bytes | str | list[bool] | np.ndarray[tuple[int], np.dtype[np.bool | np.uint8]]) -> None:
        if isinstance(value, bytes):
            length = 8 * len(value)
            data = value
        else:
            if isinstance(value, str):
                length = len(value)

                if length % 8 != 0:
                    value += '0' * (8 - (length % 8))

                try:
                    data = int(value, 2).to_bytes(len(value) // 8, byteorder='big')
                except ValueError:
                    raise ValueError('expected bit string')
            else:
                value = np.asarray(value)

                # for mypy
                assert isinstance(value, np.ndarray)

                if value.dtype != np.bool:
                    # skip warning for result of np.unpackbits
                    if value.dtype != np.uint8 or np.any(value > 1):
                        warn('expected elements to be boolean', stacklevel=2)
                    value = value.astype(bool)

                    # for mypy
                    assert isinstance(value, np.ndarray)

                if value.ndim != 1:
                    raise ValueError('expected ndim to be 1')

                length = len(value)
                data = np.packbits(value).tobytes()

        self._value = pack('>i', length) + data

    def __repr__(self) -> str:
        return f'Bit({self.to_text()})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.to_binary() == other.to_binary()
        return False

    def _length(self):
        length, = unpack_from('>i', self._value)
        return length

    def to_list(self) -> list[bool]:
        return self.to_numpy().tolist()

    def to_numpy(self) -> np.ndarray[tuple[int], np.dtype[np.bool]]:
        return np.unpackbits(np.frombuffer(self._value[4:], dtype=np.uint8), count=self._length()).astype(bool)

    def to_text(self) -> str:
        return ''.join(format(v, '08b') for v in self._value[4:])[:self._length()]

    def to_binary(self) -> bytes:
        return self._value

    @classmethod
    def from_text(cls, value: str) -> Bit:
        return cls(str(value))

    @classmethod
    def from_binary(cls, value: bytes) -> Bit:
        if not isinstance(value, bytes):
            raise ValueError('expected bytes')

        bit = cls.__new__(cls)
        bit._value = value
        return bit

    @classmethod
    def _to_db(cls, value: Bit) -> str:
        if not isinstance(value, Bit):
            raise ValueError('expected bit')

        return value.to_text()

    @classmethod
    def _to_db_binary(cls, value: Bit) -> bytes:
        if not isinstance(value, Bit):
            raise ValueError('expected bit')

        return value.to_binary()
