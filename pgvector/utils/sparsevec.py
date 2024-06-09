import numpy as np
from struct import pack, unpack_from


class SparseVector:
    def __init__(self, dim, indices, values):
        # TODO improve
        self._dim = int(dim)
        self._indices = [int(i) for i in indices]
        self._values = [float(v) for v in values]

    def __repr__(self):
        return f'SparseVector({self._dim}, {self._indices}, {self._values})'

    @classmethod
    def from_dict(cls, d, dim):
        elements = [(i, v) for i, v in d.items()]
        elements.sort()
        indices = [int(v[0]) for v in elements]
        values = [float(v[1]) for v in elements]
        return cls(dim, indices, values)

    @classmethod
    def from_sparse(cls, value):
        value = value.tocoo()

        if value.ndim == 1:
            dim = value.shape[0]
        elif value.ndim == 2 and value.shape[0] == 1:
            dim = value.shape[1]
        else:
            raise ValueError('expected ndim to be 1')

        if hasattr(value, 'coords'):
            # scipy 1.13+
            indices = value.coords[0].tolist()
        else:
            indices = value.col.tolist()
        values = value.data.tolist()
        return cls(dim, indices, values)

    @classmethod
    def from_dense(cls, value):
        dim = len(value)
        indices = [i for i, v in enumerate(value) if v != 0]
        values = [float(value[i]) for i in indices]
        return cls(dim, indices, values)

    def dim(self):
        return self._dim

    def to_dict(self):
        return dict(zip(self._indices, self._values))

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
        return '{' + ','.join([f'{int(i) + 1}:{float(v)}' for i, v in zip(self._indices, self._values)]) + '}/' + str(int(self._dim))

    def to_binary(self):
        nnz = len(self._indices)
        return pack(f'>iii{nnz}i{nnz}f', self._dim, nnz, 0, *self._indices, *self._values)

    @classmethod
    def from_text(cls, value):
        elements, dim = value.split('/')
        indices = []
        values = []
        for e in elements[1:-1].split(','):
            i, v = e.split(':')
            indices.append(int(i) - 1)
            values.append(float(v))
        return cls(int(dim), indices, values)

    @classmethod
    def from_binary(cls, value):
        dim, nnz, unused = unpack_from('>iii', value)
        indices = unpack_from(f'>{nnz}i', value, 12)
        values = unpack_from(f'>{nnz}f', value, 12 + nnz * 4)
        return cls(int(dim), indices, values)

    @classmethod
    def _to_db(cls, value, dim=None):
        if value is None:
            return value

        value = cls._to_db_value(value)

        if dim is not None and value.dim() != dim:
            raise ValueError('expected %d dimensions, not %d' % (dim, value.dim()))

        return value.to_text()

    @classmethod
    def _to_db_binary(cls, value):
        if value is None:
            return value

        value = cls._to_db_value(value)

        return value.to_binary()

    @classmethod
    def _to_db_value(cls, value):
        if isinstance(value, cls):
            return value
        elif isinstance(value, (list, np.ndarray)):
            return cls.from_dense(value)
        else:
            raise ValueError('expected sparsevec')

    @classmethod
    def _from_db(cls, value):
        if value is None or isinstance(value, cls):
            return value

        return cls.from_text(value)

    @classmethod
    def _from_db_binary(cls, value):
        if value is None or isinstance(value, cls):
            return value

        return cls.from_binary(value)
