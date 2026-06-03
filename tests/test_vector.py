from pgvector import Vector
import pytest
from struct import pack

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestVector:
    def test_list(self):
        assert Vector([1, 2, 3]).to_list() == [1, 2, 3]

    def test_list_empty(self):
        assert Vector([]).to_list() == []

    def test_list_str(self):
        with pytest.raises(ValueError) as error:
            Vector([1, 'two', 3])  # ty: ignore[invalid-argument-type]
        assert str(error.value) == 'expected list[float]'

    def test_list_list(self):
        with pytest.raises(ValueError) as error:
            Vector([[1, 2], [3, 4]])  # ty: ignore[invalid-argument-type]
        assert str(error.value) == 'expected list[float]'

    @pytest.mark.skipif(NUMPY_AVAILABLE, reason='NumPy required')
    def test_ndarray(self):
        arr = np.array([1, 2, 3])
        assert Vector(arr).to_list() == [1, 2, 3]
        assert Vector(arr).to_numpy() is not arr

    def test_int(self):
        with pytest.raises(ValueError) as error:
            Vector(1)  # ty: ignore[invalid-argument-type]
        assert str(error.value) == 'expected list or ndarray'

    def test_repr(self):
        assert repr(Vector([1, 2, 3])) == 'Vector([1.0, 2.0, 3.0])'
        assert str(Vector([1, 2, 3])) == 'Vector([1.0, 2.0, 3.0])'

    def test_equality(self):
        assert Vector([1, 2, 3]) == Vector([1, 2, 3])
        assert Vector([1, 2, 3]) != Vector([1, 2, 4])

    def test_dimensions(self):
        assert Vector([1, 2, 3]).dimensions() == 3

    @pytest.mark.skipif(NUMPY_AVAILABLE, reason='NumPy required')
    def test_to_numpy_readonly(self):
        arr = Vector([1, 2, 3]).to_numpy()
        with pytest.raises(ValueError) as error:
            arr[0] = 4
        assert str(error.value) == 'assignment destination is read-only'

    def test_from_text(self):
        vec = Vector.from_text('[1.5,2,3]')
        assert vec.to_list() == [1.5, 2, 3]
        if np is not None:
            assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])

    def test_from_binary(self):
        data = pack('>HH3f', 3, 0, 1.5, 2, 3)
        vec = Vector.from_binary(data)
        assert vec.to_list() == [1.5, 2, 3]
        if np is not None:
            assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])
        assert vec.to_binary() == data
