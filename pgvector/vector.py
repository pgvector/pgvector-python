from __future__ import annotations
import numpy as np
import struct


class Vector:
    def __init__(self, value: list[float] | np.ndarray[tuple[int], np.dtype[np.floating]]) -> None:
        if isinstance(value, list):
            dim = len(value)
            try:
                self._value = struct.pack(f'>HH{dim}f', dim, 0, *value)
            except struct.error as e:
                raise ValueError('expected list[float]')
        elif isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError('expected ndim to be 1')

            # asarray still copies if same dtype
            if value.dtype != '>f4':
                value = np.asarray(value, dtype='>f4')

            self._value = struct.pack('>HH', value.shape[0], 0) + value.tobytes()
        else:
            raise ValueError('expected list or ndarray')

    def __repr__(self) -> str:
        return f'Vector({self.to_list()})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.to_binary() == other.to_binary()
        return False

    def dimensions(self) -> int:
        dim, = struct.unpack_from('>H', self._value)
        return dim

    def to_list(self) -> list[float]:
        return list(struct.unpack_from(f'>{self.dimensions()}f', self._value[4:]))

    def to_numpy(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        return np.frombuffer(self._value, dtype='>f4', count=self.dimensions(), offset=4)

    def to_text(self) -> str:
        return '[' + ','.join([str(v) for v in self.to_list()]) + ']'

    def to_binary(self) -> bytes:
        return self._value

    @classmethod
    def from_text(cls, value: str) -> Vector:
        return cls([float(v) for v in value[1:-1].split(',')])

    @classmethod
    def from_binary(cls, value: bytes) -> Vector:
        # TODO check dimensions/length and unused
        vec = cls.__new__(cls)
        vec._value = value
        return vec

    @classmethod
    def _to_db(cls, value: object, dim: int | None = None) -> str | None:
        if value is None:
            return value

        if not isinstance(value, cls):
            value = cls(value)  # ty: ignore[invalid-argument-type]

        if dim is not None and value.dimensions() != dim:
            raise ValueError('expected %d dimensions, not %d' % (dim, value.dimensions()))

        return value.to_text()

    @classmethod
    def _to_db_binary(cls, value: object) -> bytes | None:
        if value is None:
            return value

        if not isinstance(value, cls):
            value = cls(value)  # ty: ignore[invalid-argument-type]

        return value.to_binary()

    @classmethod
    def _from_db(cls, value: str | np.ndarray | None) -> np.ndarray | None:
        if value is None or isinstance(value, np.ndarray):
            return value

        return cls.from_text(value).to_numpy().astype(np.float32)

    @classmethod
    def _from_db_binary(cls, value: bytes | np.ndarray | None) -> np.ndarray | None:
        if value is None or isinstance(value, np.ndarray):
            return value

        return cls.from_binary(value).to_numpy().astype(np.float32)
