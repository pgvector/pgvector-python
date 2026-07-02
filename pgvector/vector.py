from __future__ import annotations
import struct

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class Vector:
    def __init__(self, value: list[float] | np.ndarray[tuple[int], np.dtype[np.floating]]) -> None:
        if isinstance(value, list):
            try:
                self._value = [float(v) for v in value]
            except (TypeError, ValueError):
                raise ValueError('expected list[float]')
        elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError('expected ndim to be 1')

            self._value = [float(v) for v in value]
        else:
            raise ValueError('expected list or ndarray')

    def __repr__(self) -> str:
        return f'Vector({self._value})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self._value == other._value
        return False

    def dimensions(self) -> int:
        return len(self._value)

    def to_list(self) -> list[float]:
        return self._value

    def to_numpy(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        return np.array(self._value, dtype=np.float32)

    def to_text(self) -> str:
        return f'[{",".join([str(v) for v in self._value])}]'

    def to_binary(self) -> bytes:
        dim = len(self._value)
        return struct.pack(f'>HH{dim}f', dim, 0, *self._value)

    @classmethod
    def from_text(cls, value: str) -> Vector:
        vec = cls.__new__(cls)
        vec._value = [float(v) for v in value[1:-1].split(',')]
        return vec

    @classmethod
    def from_binary(cls, value: bytes) -> Vector:
        dim, unused = struct.unpack_from('>HH', value)

        if len(value) != 4 + 4 * dim:
            raise ValueError('invalid length')

        if unused != 0:
            raise ValueError('expected unused to be 0')

        vec = cls.__new__(cls)
        vec._value = list(struct.unpack_from(f'>{dim}f', value[4:]))
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

        # fast path for NumPy
        if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError('expected ndim to be 1')

            if value.dtype != '>f4':
                value = np.asarray(value, dtype='>f4')

            return struct.pack('>HH', len(value), 0) + value.tobytes()

        if not isinstance(value, cls):
            value = cls(value)  # type: ignore

        return value.to_binary()

    @classmethod
    def _from_db(cls, value: str | Vector | None) -> list[float] | None:
        if value is None:
            return value

        if isinstance(value, str):
            value = cls.from_text(value)

        return value.to_list()
