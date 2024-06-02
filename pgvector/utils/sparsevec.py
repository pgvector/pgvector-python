import numpy as np
from struct import pack, unpack_from


def to_db_value(value):
    if isinstance(value, SparseVector):
        return value
    elif isinstance(value, (list, np.ndarray)):
        return SparseVector.from_dense(value)
    else:
        raise ValueError('expected sparsevec')


class SparseVector:
    def __init__(self, dim, indices, values):
        self._dim = dim
        self._indices = indices
        self._values = values

    def __repr__(self):
        return f'SparseVector({self._dim}, {self._indices}, {self._values})'

    def from_dense(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        dim = len(value)
        indices = [i for i, v in enumerate(value) if v != 0]
        values = [value[i] for i in indices]
        return SparseVector(dim, indices, values)

    def dim(self):
        return self._dim

    def to_list(self):
        vec = [0.0] * self._dim
        for i, v in zip(self._indices, self._values):
            vec[i] = v
        return vec

    def to_numpy(self):
        vec = np.repeat(0.0, self._dim).astype(np.float32)
        for i, v in zip(self._indices, self._values):
            vec[i] = v
        return vec

    def to_text(self):
        return '{' + ','.join([f'{i + 1}:{v}' for i, v in zip(self._indices, self._values)]) + '}/' + str(self._dim)

    def to_binary(self):
        nnz = len(self._indices)
        return pack(f'>iii{nnz}i{nnz}f', self._dim, nnz, 0, *self._indices, *self._values)

    def from_text(value):
        elements, dim = value.split('/')
        indices = []
        values = []
        for e in elements[1:-1].split(','):
            i, v = e.split(':')
            indices.append(int(i) - 1)
            values.append(float(v))
        return SparseVector(int(dim), indices, values)

    def from_binary(value):
        dim, nnz, unused = unpack_from('>iii', value)
        indices = unpack_from(f'>{nnz}i', value, 12)
        values = unpack_from(f'>{nnz}f', value, 12 + nnz * 4)
        return SparseVector(int(dim), indices, values)

    # TODO move rest

    def to_db(value, dim=None):
        if value is None:
            return value

        value = to_db_value(value)

        if dim is not None and value.dim() != dim:
            raise ValueError('expected %d dimensions, not %d' % (dim, value.dim()))

        return value.to_text()

    def to_db_binary(value):
        if value is None:
            return value

        value = to_db_value(value)

        return value.to_binary()

    def from_db(value):
        if value is None or isinstance(value, SparseVector):
            return value

        return SparseVector.from_text(value)

    def from_db_binary(value):
        if value is None or isinstance(value, SparseVector):
            return value

        return SparseVector.from_binary(value)
