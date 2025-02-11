import numpy as np
from pgvector import SparseVector
import pytest
from scipy.sparse import coo_array
from struct import pack


class TestSparseVector:
    def test_list(self):
        vec = SparseVector([1, 0, 2, 0, 3, 0])
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert np.array_equal(vec.to_numpy(), [1, 0, 2, 0, 3, 0])
        assert vec.indices() == [0, 2, 4]

    def test_list_dimensions(self):
        with pytest.raises(ValueError) as error:
            SparseVector([1, 0, 2, 0, 3, 0], 6)
        assert str(error.value) == 'extra argument'

    def test_ndarray(self):
        vec = SparseVector(np.array([1, 0, 2, 0, 3, 0]))
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_dict(self):
        vec = SparseVector({2: 2, 4: 3, 0: 1, 3: 0}, 6)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_dict_no_dimensions(self):
        with pytest.raises(ValueError) as error:
            SparseVector({0: 1, 2: 2, 4: 3})
        assert str(error.value) == 'missing dimensions'

    def test_coo_array(self):
        arr = coo_array(np.array([1, 0, 2, 0, 3, 0]))
        vec = SparseVector(arr)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_coo_array_dimensions(self):
        with pytest.raises(ValueError) as error:
            SparseVector(coo_array(np.array([1, 0, 2, 0, 3, 0])), 6)
        assert str(error.value) == 'extra argument'

    def test_dok_array(self):
        arr = coo_array(np.array([1, 0, 2, 0, 3, 0])).todok()
        vec = SparseVector(arr)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_repr(self):
        assert repr(SparseVector([1, 0, 2, 0, 3, 0])) == 'SparseVector({0: 1.0, 2: 2.0, 4: 3.0}, 6)'
        assert str(SparseVector([1, 0, 2, 0, 3, 0])) == 'SparseVector({0: 1.0, 2: 2.0, 4: 3.0}, 6)'

    def test_equality(self):
        assert SparseVector([1, 0, 2, 0, 3, 0]) == SparseVector([1, 0, 2, 0, 3, 0])
        assert SparseVector([1, 0, 2, 0, 3, 0]) != SparseVector([1, 0, 2, 0, 3, 1])
        assert SparseVector([1, 0, 2, 0, 3, 0]) == SparseVector({2: 2, 4: 3, 0: 1, 3: 0}, 6)
        assert SparseVector({}, 1) != SparseVector({}, 2)

    def test_dimensions(self):
        assert SparseVector([1, 0, 2, 0, 3, 0]).dimensions() == 6

    def test_indices(self):
        assert SparseVector([1, 0, 2, 0, 3, 0]).indices() == [0, 2, 4]

    def test_values(self):
        assert SparseVector([1, 0, 2, 0, 3, 0]).values() == [1, 2, 3]

    def test_to_coo(self):
        assert np.array_equal(SparseVector([1, 0, 2, 0, 3, 0]).to_coo().toarray(), [[1, 0, 2, 0, 3, 0]])

    def test_zero_vector_text(self):
        vec = SparseVector({}, 3)
        assert vec.to_list() == SparseVector.from_text(vec.to_text()).to_list()

    def test_from_text(self):
        vec = SparseVector.from_text('{1:1.5,3:2,5:3}/6')
        assert vec.dimensions() == 6
        assert vec.indices() == [0, 2, 4]
        assert vec.values() == [1.5, 2, 3]
        assert vec.to_list() == [1.5, 0, 2, 0, 3, 0]
        assert np.array_equal(vec.to_numpy(), [1.5, 0, 2, 0, 3, 0])

    def test_from_binary(self):
        data = pack('>iii3i3f', 6, 3, 0, 0, 2, 4, 1.5, 2, 3)
        vec = SparseVector.from_binary(data)
        assert vec.dimensions() == 6
        assert vec.indices() == [0, 2, 4]
        assert vec.values() == [1.5, 2, 3]
        assert vec.to_list() == [1.5, 0, 2, 0, 3, 0]
        assert np.array_equal(vec.to_numpy(), [1.5, 0, 2, 0, 3, 0])
        assert vec.to_binary() == data
