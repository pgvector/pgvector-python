import numpy as np
from struct import pack, unpack_from


def to_db_value(value):
    if isinstance(value, SparseVec):
        return value
    elif isinstance(value, (list, np.ndarray)):
        return SparseVec.from_dense(value)
    else:
        raise ValueError('expected sparsevec')


class SparseVec:
    def __init__(self, dim, indices, values):
        self.dim = dim
        self.indices = indices
        self.values = values

    def from_dense(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        dim = len(value)
        indices = [i for i, v in enumerate(value) if v != 0]
        values = [value[i] for i in indices]
        return SparseVec(dim, indices, values)

    def to_dense(self):
        vec = [0] * self.dim
        for i, v in zip(self.indices, self.values):
            vec[i] = v
        return vec

    def to_db(value, dim=None):
        if value is None:
            return value

        value = to_db_value(value)

        if dim is not None and value.dim != dim:
            raise ValueError('expected %d dimensions, not %d' % (dim, value.dim))

        return '{' + ','.join([f'{i + 1}:{v}' for i, v in zip(value.indices, value.values)]) + '}/' + str(value.dim)

    def to_db_binary(value):
        if value is None:
            return value

        value = to_db_value(value)
        nnz = len(value.indices)
        return pack(f'>iii{nnz}i{nnz}f', value.dim, nnz, 0, *value.indices, *value.values)

    def from_db(value):
        if value is None or isinstance(value, SparseVec):
            return value
        elements, dim = value.split('/')
        indices = []
        values = []
        for e in elements[1:-1].split(','):
            i, v = e.split(':')
            indices.append(int(i) - 1)
            values.append(float(v))
        return SparseVec(int(dim), indices, values)

    def from_db_binary(value):
        if value is None or isinstance(value, SparseVec):
            return value
        dim, nnz, unused = unpack_from('>iii', value)
        indices = list(unpack_from(f'>{nnz}i', value, 12))
        values = list(unpack_from(f'>{nnz}f', value, 12 + nnz * 4))
        return SparseVec(int(dim), indices, values)

    def __repr__(self):
        return f'SparseVec({self.dim}, {self.indices}, {self.values})'
