import numpy as np
from pgvector.utils import SparseVector
import pytest


class TestSparseVector:
    def test_from_dense(self):
        assert SparseVector.from_dense([1, 0, 2, 0, 3, 0]).to_list() == [1, 0, 2, 0, 3, 0]
        assert SparseVector.from_dense([1, 0, 2, 0, 3, 0]).to_numpy().tolist() == [1, 0, 2, 0, 3, 0]

    def test_from_coordinates(self):
        assert SparseVector.from_coordinates({0: 1, 2: 2, 4: 3}, 6).to_list() == [1, 0, 2, 0, 3, 0]

    def test_repr(self):
        assert repr(SparseVector.from_dense([1, 0, 2, 0, 3, 0])) == 'SparseVector(6, [0, 2, 4], [1.0, 2.0, 3.0])'
        assert str(SparseVector.from_dense([1, 0, 2, 0, 3, 0])) == 'SparseVector(6, [0, 2, 4], [1.0, 2.0, 3.0])'

    def test_dim(self):
        assert SparseVector.from_dense([1, 0, 2, 0, 3, 0]).dim() == 6
