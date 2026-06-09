from __future__ import annotations
from struct import pack, unpack_from
from typing import Any, overload

try:
    import numpy as np
except ImportError:
    pass

NO_DEFAULT = object()


class SparseVector:
    @overload
    def __init__(self, value: dict[int, float], dimensions: int, /) -> None:
        ...

    @overload
    def __init__(self, value: list[float], /) -> None:
        ...

    @overload
    def __init__(self, value: Any, /) -> None:
        ...

    def __init__(self, value: dict[int, float] | list[float] | Any, dimensions: int | Any = NO_DEFAULT, /) -> None:
        if value.__class__.__module__.startswith('scipy.sparse.'):
            if dimensions is not NO_DEFAULT:
                raise ValueError('extra argument')

            self._from_sparse(value)
        elif isinstance(value, dict):
            if dimensions is NO_DEFAULT:
                raise ValueError('missing dimensions')

            self._from_dict(value, dimensions)
        else:
            if dimensions is not NO_DEFAULT:
                raise ValueError('extra argument')

            self._from_dense(value)

    def __repr__(self) -> str:
        elements = dict(zip(self._indices, self._values))
        return f'SparseVector({elements}, {self._dim})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.dimensions() == other.dimensions() and self.indices() == other.indices() and self.values() == other.values()
        return False

    def dimensions(self) -> int:
        return self._dim

    def indices(self) -> list[int]:
        return self._indices

    def values(self) -> list[float]:
        return self._values

    def to_coo(self) -> Any:
        from scipy.sparse import coo_array

        coords = ([0] * len(self._indices), self._indices)
        return coo_array((self._values, coords), shape=(1, self._dim))

    def to_list(self) -> list[float]:
        vec = [0.0] * self._dim
        for i, v in zip(self._indices, self._values):
            vec[i] = v
        return vec

    def to_numpy(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        vec = np.repeat(0.0, self._dim).astype(np.float32)
        for i, v in zip(self._indices, self._values):
            vec[i] = v
        return vec

    def to_text(self) -> str:
        return '{' + ','.join([f'{int(i) + 1}:{float(v)}' for i, v in zip(self._indices, self._values)]) + '}/' + str(int(self._dim))

    def to_binary(self) -> bytes:
        nnz = len(self._indices)
        return pack(f'>iii{nnz}i{nnz}f', self._dim, nnz, 0, *self._indices, *self._values)

    def _from_dict(self, d: dict[int, float], dim: int) -> None:
        elements = [(i, v) for i, v in d.items() if v != 0]
        elements.sort()

        self._dim = int(dim)
        self._indices = [int(v[0]) for v in elements]
        self._values = [float(v[1]) for v in elements]

    def _from_sparse(self, value: Any) -> None:
        value = value.tocoo()

        if value.ndim == 1:
            self._dim = value.shape[0]
        elif value.ndim == 2 and value.shape[0] == 1:
            self._dim = value.shape[1]
        else:
            raise ValueError('expected ndim to be 1')

        if hasattr(value, 'coords'):
            # scipy 1.13+
            self._indices = value.coords[-1].tolist()
        else:
            self._indices = value.col.tolist()
        self._values = value.data.tolist()

    def _from_dense(self, value: list[float]) -> None:
        self._dim = len(value)
        self._indices = [i for i, v in enumerate(value) if v != 0]
        self._values = [float(value[i]) for i in self._indices]

    @classmethod
    def from_text(cls, value: str) -> SparseVector:
        elements, dim = value.split('/', 2)
        indices = []
        values = []
        # split on empty string returns single element list
        if len(elements) > 2:
            for e in elements[1:-1].split(','):
                i, v = e.split(':', 2)
                indices.append(int(i) - 1)
                values.append(float(v))
        return cls._from_parts(int(dim), indices, values)

    @classmethod
    def from_binary(cls, value: bytes) -> SparseVector:
        dim, nnz, unused = unpack_from('>iii', value)

        if len(value) != 12 + 8 * nnz:
            raise ValueError('invalid length')

        if unused != 0:
            raise ValueError('expected unused to be 0')

        indices = unpack_from(f'>{nnz}i', value, 12)
        values = unpack_from(f'>{nnz}f', value, 12 + nnz * 4)
        return cls._from_parts(int(dim), list(indices), list(values))

    @classmethod
    def _from_parts(cls, dim: int, indices: list[int], values: list[float]) -> SparseVector:
        vec = cls.__new__(cls)
        vec._dim = dim
        vec._indices = indices
        vec._values = values
        return vec

    @classmethod
    def _to_db(cls, value: object) -> str | None:
        if value is None:
            return value

        if not isinstance(value, cls):
            value = cls(value)

        return value.to_text()

    @classmethod
    def _to_db_binary(cls, value: object) -> bytes | None:
        if value is None:
            return value

        if not isinstance(value, cls):
            value = cls(value)

        return value.to_binary()

    @classmethod
    def _from_db(cls, value: str | SparseVector | None) -> SparseVector | None:
        if value is None or isinstance(value, SparseVector):
            return value

        return cls.from_text(value)

    @classmethod
    def _from_db_binary(cls, value: bytes | SparseVector | None) -> SparseVector | None:
        if value is None or isinstance(value, SparseVector):
            return value

        return cls.from_binary(value)
