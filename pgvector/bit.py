from __future__ import annotations
import numpy as np
from struct import pack, unpack_from
from typing import Any
from warnings import warn


class Bit:
    def __init__(self, value: Any) -> None:
        if isinstance(value, bytes):
            self._len = 8 * len(value)
            self._data = value
        else:
            if isinstance(value, str):
                value = [v != '0' for v in value]
            else:
                value = np.asarray(value)

                if value.dtype != np.bool:
                    # skip warning for result of np.unpackbits
                    if value.dtype != np.uint8 or np.any(value > 1):
                        warn('expected elements to be boolean', stacklevel=2)
                    value = value.astype(bool)

                if value.ndim != 1:
                    raise ValueError('expected ndim to be 1')

            self._len = len(value)
            self._data = np.packbits(value).tobytes()

    def __repr__(self) -> str:
        return f'Bit({self.to_text()})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self._len == other._len and self._data == other._data
        return False

    def to_list(self) -> list[bool]:
        return self.to_numpy().tolist()

    def to_numpy(self) -> np.ndarray:
        return np.unpackbits(np.frombuffer(self._data, dtype=np.uint8), count=self._len).astype(bool)

    def to_text(self) -> str:
        return ''.join(format(v, '08b') for v in self._data)[:self._len]

    def to_binary(self) -> bytes:
        return pack('>i', self._len) + self._data

    @classmethod
    def from_text(cls, value: str) -> Bit:
        return cls(str(value))

    @classmethod
    def from_binary(cls, value: bytes) -> Bit:
        if not isinstance(value, bytes):
            raise ValueError('expected bytes')

        bit = cls.__new__(cls)
        bit._len = unpack_from('>i', value)[0]
        bit._data = value[4:]
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
