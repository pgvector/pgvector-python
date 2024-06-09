import numpy as np
from pgvector.utils import SparseVector
import pytest
from scipy.sparse import coo_array


class TestSparseVector:
    def test_from_dense(self):
        assert SparseVector.from_dense([1, 0, 2, 0, 3, 0]).to_list() == [1, 0, 2, 0, 3, 0]
        assert SparseVector.from_dense([1, 0, 2, 0, 3, 0]).to_numpy().tolist() == [1, 0, 2, 0, 3, 0]
        assert SparseVector.from_dense(np.array([1, 0, 2, 0, 3, 0])).to_list() == [1, 0, 2, 0, 3, 0]

    def test_from_coordinates(self):
        assert SparseVector.from_dict({0: 1, 2: 2, 4: 3}, 6).to_list() == [1, 0, 2, 0, 3, 0]

    def test_from_sparse(self):
        arr = coo_array(np.array([1, 0, 2, 0, 3, 0]))
        assert SparseVector.from_sparse(arr).to_list() == [1, 0, 2, 0, 3, 0]
        assert SparseVector.from_sparse(arr.tocsc()).to_list() == [1, 0, 2, 0, 3, 0]
        assert SparseVector.from_sparse(arr.tocsr()).to_list() == [1, 0, 2, 0, 3, 0]
        assert SparseVector.from_sparse(arr.todok()).to_list() == [1, 0, 2, 0, 3, 0]

    def test_repr(self):
        assert repr(SparseVector.from_dense([1, 0, 2, 0, 3, 0])) == 'SparseVector(6, [0, 2, 4], [1.0, 2.0, 3.0])'
        assert str(SparseVector.from_dense([1, 0, 2, 0, 3, 0])) == 'SparseVector(6, [0, 2, 4], [1.0, 2.0, 3.0])'

    def test_dim(self):
        assert SparseVector.from_dense([1, 0, 2, 0, 3, 0]).dim() == 6
