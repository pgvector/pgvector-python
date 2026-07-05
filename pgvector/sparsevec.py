from __future__ import annotations
from struct import pack, unpack_from
from typing import TYPE_CHECKING, Final, cast, overload
from ._utils import is_sparse_array, ndarray

if TYPE_CHECKING:
    import numpy as np
    from scipy.sparse import sparray, spmatrix, coo_array, coo_matrix


class Sentinel:
    pass


NO_DEFAULT: Final[Sentinel] = Sentinel()


class SparseVector:
    _dim: int
    _indices: list[int]
    _values: list[float]

    @overload
    def __init__(self, value: dict[int, float], dimensions: int, /) -> None: ...

    @overload
    def __init__(self, value: list[float] | ndarray | sparray | spmatrix, /) -> None: ...

    def __init__(
        self,
        value: dict[int, float] | list[float] | ndarray | sparray | spmatrix,
        dimensions: int | Sentinel = NO_DEFAULT,
        /
    ) -> None:
        if is_sparse_array(value):
            if dimensions is not NO_DEFAULT:
                raise ValueError('extra argument')

            self._from_sparse(value)  # type: ignore
        elif isinstance(value, dict):
            if dimensions is NO_DEFAULT:
                raise ValueError('missing dimensions')

            self._from_dict(value, dimensions)  # type: ignore
        else:
            if dimensions is not NO_DEFAULT:
                raise ValueError('extra argument')

            self._from_dense(value)  # type: ignore

    def __repr__(self) -> str:
        elements = dict(zip(self._indices, self._values))
        return f'SparseVector({elements}, {self._dim})'

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._dim == other._dim and self._indices == other._indices and self._values == other._values

    def dimensions(self) -> int:
        return self._dim

    def indices(self) -> list[int]:
        return self._indices

    def values(self) -> list[float]:
        return self._values

    def to_coo(self) -> coo_array:
        from scipy.sparse import coo_array

        coords = ([0] * len(self._indices), self._indices)
        return coo_array((self._values, coords), shape=(1, self._dim))

    def to_list(self) -> list[float]:
        vec = [0.0] * self._dim
        for i, v in zip(self._indices, self._values):
            vec[i] = v
        return vec

    def to_numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        import numpy as np

        vec = np.zeros(self._dim, dtype=np.float32)
        for i, v in zip(self._indices, self._values):
            vec[i] = v
        return vec

    def to_text(self) -> str:
        elements = ','.join([f'{int(i) + 1}:{float(v)}' for i, v in zip(self._indices, self._values)])
        return f'{{{elements}}}/{int(self._dim)}'

    def to_binary(self) -> bytes:
        nnz = len(self._indices)
        return pack(f'>iii{nnz}i{nnz}f', self._dim, nnz, 0, *self._indices, *self._values)

    def _from_dict(self, d: dict[int, float], dim: int) -> None:
        elements = [(i, v) for i, v in d.items() if v != 0]
        elements.sort()

        self._dim = int(dim)
        self._indices = [int(v[0]) for v in elements]
        self._values = [float(v[1]) for v in elements]

    def _from_sparse(self, arr: sparray | spmatrix, /) -> None:
        value: coo_array | coo_matrix = arr.tocoo(copy=False)  # type: ignore

        shape = cast(tuple[int, ...], value.shape)
        if len(shape) == 1:
            self._dim = shape[0]
        elif len(shape) == 2 and shape[0] == 1:
            self._dim = shape[1]
        else:
            raise ValueError('expected ndim to be 1')

        if hasattr(value, 'coords'):
            # scipy 1.13+
            self._indices = value.coords[-1].tolist()
        else:
            self._indices = value.col.tolist()
        self._values = [float(v) for v in value.data]

    def _from_dense(self, value: list[float] | ndarray, /) -> None:
        self._dim = len(value)
        self._indices = [i for i, v in enumerate(value) if v != 0]
        self._values = [float(value[i]) for i in self._indices]

    @classmethod
    def from_text(cls, value: str, /) -> SparseVector:
        elements, dim = value.split('/', 2)
        indices: list[int] = []
        values: list[float] = []
        # split on empty string returns single element list
        if len(elements) > 2:
            for e in elements[1:-1].split(','):
                i, v = e.split(':', 2)
                indices.append(int(i) - 1)
                values.append(float(v))
        return cls._from_parts(int(dim), indices, values)

    @classmethod
    def from_binary(cls, value: bytes | bytearray | memoryview, /) -> SparseVector:
        dim, nnz, unused = unpack_from('>iii', value)

        if len(value) != 12 + 8 * nnz:
            raise ValueError('invalid length')

        if unused != 0:
            raise ValueError('expected unused to be 0')

        indices = list(unpack_from(f'>{nnz}i', value, 12))
        values = list(unpack_from(f'>{nnz}f', value, 12 + nnz * 4))
        return cls._from_parts(dim, indices, values)

    @classmethod
    def _from_parts(cls, dim: int, indices: list[int], values: list[float], /) -> SparseVector:
        vec = cls.__new__(cls)
        vec._dim = dim
        vec._indices = indices
        vec._values = values
        return vec

    @classmethod
    def _to_db(cls, value: list[float] | ndarray | sparray | spmatrix | SparseVector | None, /) -> str | None:
        if value is None:
            return value

        if not isinstance(value, SparseVector):
            value = cls(value)

        return value.to_text()

    @classmethod
    def _from_db(cls, value: str | SparseVector | None, /) -> SparseVector | None:
        if value is None or isinstance(value, SparseVector):
            return value

        return cls.from_text(value)
