from __future__ import annotations
import array
import struct
import sys

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class HalfVector:
    def __init__(self, value: list[float] | np.ndarray[tuple[int], np.dtype[np.floating]]) -> None:
        if isinstance(value, list):
            dim = len(value)
            try:
                self._value = array.array('H', struct.pack(f'{dim}e', *value))
            except struct.error:
                raise ValueError('expected list[float]')
        elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError('expected ndim to be 1')

            if value.dtype != np.float16:
                value = np.asarray(value, dtype=np.float16)

            # tobytes() important for performance
            self._value = array.array('H', value.tobytes())
        else:
            raise ValueError('expected list or ndarray')

    def __repr__(self) -> str:
        return f'HalfVector({self.to_list()})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self._value == other._value
        return False

    def dimensions(self) -> int:
        return len(self._value)

    def to_list(self) -> list[float]:
        dim = len(self._value)
        return list(struct.unpack(f'{dim}e', self._value.tobytes()))

    def to_numpy(self) -> np.ndarray[tuple[int], np.dtype[np.float16]]:
        return np.frombuffer(self._value, dtype=np.float16)

    def to_text(self) -> str:
        return f'[{",".join([str(v) for v in self.to_list()])}]'

    def to_binary(self) -> bytes:
        if sys.byteorder == 'big':
            value = self._value
        else:
            value = array.array('H', self._value)
            value.byteswap()
        return struct.pack(f'>HH', len(value), 0) + value.tobytes()

    @classmethod
    def from_text(cls, value: str) -> HalfVector:
        return cls([float(v) for v in value[1:-1].split(',')])

    @classmethod
    def from_binary(cls, value: bytes) -> HalfVector:
        dim, unused = struct.unpack_from('>HH', value)

        if len(value) != 4 + 2 * dim:
            raise ValueError('invalid length')

        if unused != 0:
            raise ValueError('expected unused to be 0')

        vec = cls.__new__(cls)
        vec._value = array.array('H', value[4:])
        if sys.byteorder != 'big':
            vec._value.byteswap()
        return vec

    @classmethod
    def _to_db(cls, value: object) -> str | None:
        if value is None:
            return value

        # fast path for high-level libraries
        if isinstance(value, list):
            return f'[{",".join([str(float(v)) for v in value])}]'  # type: ignore

        if not isinstance(value, cls):
            value = cls(value)  # type: ignore

        return value.to_text()

    @classmethod
    def _to_db_binary(cls, value: object) -> bytes | None:
        if value is None:
            return value

        if not isinstance(value, cls):
            value = cls(value)  # type: ignore

        return value.to_binary()

    @classmethod
    def _from_db(cls, value: str | HalfVector | None) -> list[float] | None:
        if value is None:
            return value

        if isinstance(value, str):
            value = cls.from_text(value)

        return value.to_list()
