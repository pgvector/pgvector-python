from struct import pack, unpack_from


class SparseVec:
    def __init__(self, dim, indices, values):
        self.dim = dim
        self.indices = indices
        self.values = values

    def from_dense(value):
        dim = len(value)
        indices = [i for i, v in enumerate(value) if v != 0]
        values = [value[i] for i in indices]
        return SparseVec(dim, indices, values)

    def to_dense(self):
        vec = [0] * self.dim
        for i, v in zip(self.indices, self.values):
            vec[i] = v
        return vec

    def to_db(self):
        return '{' + ','.join([f'{i + 1}:{v}' for i, v in zip(self.indices, self.values)]) + '}/' + str(self.dim)

    def to_db_binary(self):
        nnz = len(self.indices)
        return pack(f'>iii{nnz}i{nnz}f', self.dim, nnz, 0, *self.indices, *self.values)

    def from_db(value):
        elements, dim = value.split('/')
        indices = []
        values = []
        for e in elements[1:-1].split(','):
            i, v = e.split(':')
            indices.append(int(i) - 1)
            values.append(float(v))
        return SparseVec(int(dim), indices, values)

    def from_db_binary(value):
        dim, nnz, unused = unpack_from('>iii', value)
        indices = unpack_from(f'>{nnz}i', value, 12)
        values = unpack_from(f'>{nnz}f', value, 12 + nnz * 4)
        return SparseVec(int(dim), indices, values)

    def __repr__(self):
        return f'SparseVec({self.dim}, {self.indices}, {self.values})'
