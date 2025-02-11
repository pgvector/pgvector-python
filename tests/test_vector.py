import numpy as np
from pgvector import Vector
import pytest
from struct import pack


class TestVector:
    def test_list(self):
        assert Vector([1, 2, 3]).to_list() == [1, 2, 3]

    def test_list_str(self):
        with pytest.raises(ValueError, match='could not convert string to float'):
            Vector([1, 'two', 3])

    def test_tuple(self):
        assert Vector((1, 2, 3)).to_list() == [1, 2, 3]

    def test_ndarray(self):
        arr = np.array([1, 2, 3])
        assert Vector(arr).to_list() == [1, 2, 3]
        assert Vector(arr).to_numpy() is not arr

    def test_ndarray_same_object(self):
        arr = np.array([1, 2, 3], dtype='>f4')
        assert Vector(arr).to_list() == [1, 2, 3]
        assert Vector(arr).to_numpy() is arr

    def test_ndim_two(self):
        with pytest.raises(ValueError) as error:
            Vector([[1, 2], [3, 4]])
        assert str(error.value) == 'expected ndim to be 1'

    def test_ndim_zero(self):
        with pytest.raises(ValueError) as error:
            Vector(1)
        assert str(error.value) == 'expected ndim to be 1'

    def test_repr(self):
        assert repr(Vector([1, 2, 3])) == 'Vector([1.0, 2.0, 3.0])'
        assert str(Vector([1, 2, 3])) == 'Vector([1.0, 2.0, 3.0])'

    def test_equality(self):
        assert Vector([1, 2, 3]) == Vector([1, 2, 3])
        assert Vector([1, 2, 3]) != Vector([1, 2, 4])

    def test_dimensions(self):
        assert Vector([1, 2, 3]).dimensions() == 3

    def test_from_text(self):
        vec = Vector.from_text('[1.5,2,3]')
        assert vec.to_list() == [1.5, 2, 3]
        assert np.array_equal(vec.to_numpy(), np.array([1.5, 2, 3]))

    def test_from_binary(self):
        data = pack('>HH3f', 3, 0, 1.5, 2, 3)
        vec = Vector.from_binary(data)
        assert vec.to_list() == [1.5, 2, 3]
        assert np.array_equal(vec.to_numpy(), np.array([1.5, 2, 3]))
        assert vec.to_binary() == data
