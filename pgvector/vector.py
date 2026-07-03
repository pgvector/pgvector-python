from __future__ import annotations
from array import array
import struct
import sys
from typing import TYPE_CHECKING
from ._utils import is_ndarray, ndarray

if TYPE_CHECKING:
    import numpy as np


class Vector:
    _value: array[float]

    def __init__(self, value: list[float] | ndarray, /) -> None:
        if isinstance(value, list):
            try:
                self._value = array('f', value)
            except TypeError:
                raise ValueError('expected list[float]')
        elif is_ndarray(value):
            import numpy as np

            if value.ndim != 1:
                raise ValueError('expected ndim to be 1')

            arr = array('f')
            arr.frombytes(value.astype(np.float32, order='C', copy=False).data.cast('B'))
            self._value = arr
        else:
            raise ValueError('expected list or ndarray')

    def __repr__(self) -> str:
        return f'Vector({self.to_list()})'

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._value == other._value

    def dimensions(self) -> int:
        return len(self._value)

    def to_list(self) -> list[float]:
        return self._value.tolist()

    def to_numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        import numpy as np
        return np.frombuffer(self._value, dtype=np.float32)

    def to_text(self) -> str:
        return f'[{",".join([str(v) for v in self._value])}]'

    def to_binary(self) -> bytes:
        if sys.byteorder == 'big':
            value = self._value
        else:
            value = array('f', self._value)
            value.byteswap()
        return struct.pack('>HH', len(value), 0) + memoryview(value)

    @classmethod
    def from_text(cls, value: str, /) -> Vector:
        return cls(cls._from_text(value))

    @classmethod
    def from_binary(cls, value: bytes | bytearray | memoryview, /) -> Vector:
        dim, unused = struct.unpack_from('>HH', value)
        data = memoryview(value)[4:]

        if len(data) != 4 * dim:
            raise ValueError('invalid length')

        if unused != 0:
            raise ValueError('expected unused to be 0')

        arr = array('f')
        arr.frombytes(data)
        if sys.byteorder != 'big':
            arr.byteswap()

        vec = cls.__new__(cls)
        vec._value = arr
        return vec

    @classmethod
    def _from_text(cls, value: str, /) -> list[float]:
        return [float(v) for v in value[1:-1].split(',')]

    @classmethod
    def _to_db(cls, value: list[float] | ndarray | Vector | None, /) -> str | None:
        if value is None:
            return value

        # fewer allocations for lists
        if isinstance(value, list):
            return f'[{",".join([str(float(v)) for v in value])}]'  # type: ignore

        if not isinstance(value, Vector):
            value = cls(value)

        return value.to_text()

    @classmethod
    def _from_db(cls, value: str | Vector | None, /) -> list[float] | None:
        if value is None:
            return value

        if isinstance(value, Vector):
            return value.to_list()

        return cls._from_text(value)
