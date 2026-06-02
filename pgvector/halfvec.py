from __future__ import annotations
import numpy as np
from struct import pack, unpack_from
from typing import Any


class HalfVector:
    def __init__(self, value: Any) -> None:
        # asarray still copies if same dtype
        if not isinstance(value, np.ndarray) or value.dtype != '>f2':
            value = np.asarray(value, dtype='>f2')

        if value.ndim != 1:
            raise ValueError('expected ndim to be 1')

        self._value = value

    def __repr__(self) -> str:
        return f'HalfVector({self.to_list()})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return np.array_equal(self.to_numpy(), other.to_numpy())
        return False

    def dimensions(self) -> int:
        return len(self._value)

    def to_list(self) -> list[float]:
        return self._value.tolist()

    def to_numpy(self) -> np.ndarray:
        return self._value

    def to_text(self) -> str:
        return '[' + ','.join([str(float(v)) for v in self._value]) + ']'

    def to_binary(self) -> bytes:
        return pack('>HH', self.dimensions(), 0) + self._value.tobytes()

    @classmethod
    def from_text(cls, value: str) -> HalfVector:
        return cls([float(v) for v in value[1:-1].split(',')])

    @classmethod
    def from_binary(cls, value: bytes) -> HalfVector:
        dim, unused = unpack_from('>HH', value)
        return cls(np.frombuffer(value, dtype='>f2', count=dim, offset=4))

    @classmethod
    def _to_db(cls, value: object, dim: int | None = None) -> str | None:
        if value is None:
            return value

        if not isinstance(value, cls):
            value = cls(value)

        if dim is not None and value.dimensions() != dim:
            raise ValueError('expected %d dimensions, not %d' % (dim, value.dimensions()))

        return value.to_text()

    @classmethod
    def _to_db_binary(cls, value: object) -> bytes | None:
        if value is None:
            return value

        if not isinstance(value, cls):
            value = cls(value)

        return value.to_binary()

    @classmethod
    def _from_db(cls, value: str | HalfVector | None) -> HalfVector | None:
        if value is None or isinstance(value, HalfVector):
            return value

        return cls.from_text(value)

    @classmethod
    def _from_db_binary(cls, value: bytes | HalfVector | None) -> HalfVector | None:
        if value is None or isinstance(value, HalfVector):
            return value

        return cls.from_binary(value)
