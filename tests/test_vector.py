from pgvector import Vector
import pytest
from struct import pack

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestVector:
    def test_list(self) -> None:
        arr = [1.0, 2.0, 3.0]
        assert Vector(arr).to_list() == arr
        assert Vector(arr).to_list() is not arr

    def test_list_empty(self) -> None:
        assert Vector([]).to_list() == []

    def test_list_str(self) -> None:
        with pytest.raises(ValueError) as error:
            Vector([1, 'two', 3])  # type: ignore
        assert str(error.value) == 'expected list[float]'

    def test_list_list(self) -> None:
        with pytest.raises(ValueError) as error:
            Vector([[1, 2], [3, 4]])  # type: ignore
        assert str(error.value) == 'expected list[float]'

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason='NumPy required')
    def test_ndarray(self) -> None:
        arr = np.array([1, 2, 3], dtype=np.float32)
        assert Vector(arr).to_list() == [1, 2, 3]
        assert Vector(arr).to_numpy() is not arr
        assert Vector(arr).to_numpy().dtype == np.float32
        # non-contiguous
        assert Vector(np.flip(arr)).to_list() == [3, 2, 1]
        assert Vector(np.flip(arr)).to_binary() == Vector([3, 2, 1]).to_binary()
        # big endian
        assert Vector(arr.astype('>f4')).to_list() == [1, 2, 3]
        assert Vector(arr.astype('>f4')).to_binary() == Vector([1, 2, 3]).to_binary()

        with pytest.raises(ValueError) as error:
            Vector(np.array(['one', 'two', 'three']))
        assert 'could not convert string to float' in str(error.value)

    def test_int(self) -> None:
        with pytest.raises(ValueError) as error:
            Vector(1)  # type: ignore
        assert str(error.value) == 'expected list or ndarray'

    def test_repr(self) -> None:
        assert repr(Vector([1, 2, 3])) == 'Vector([1.0, 2.0, 3.0])'
        assert str(Vector([1, 2, 3])) == 'Vector([1.0, 2.0, 3.0])'

    def test_equality(self) -> None:
        assert Vector([1, 2, 3]) == Vector([1, 2, 3])
        assert Vector([1, 2, 3]) != Vector([1, 2, 4])

    def test_dimensions(self) -> None:
        assert Vector([1, 2, 3]).dimensions() == 3

    def test_from_text(self) -> None:
        vec = Vector.from_text('[1.5,2,3]')
        assert vec.to_list() == [1.5, 2, 3]
        if NUMPY_AVAILABLE:
            assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])

    def test_from_binary(self) -> None:
        data = pack('>HH3f', 3, 0, 1.5, 2, 3)
        vec = Vector.from_binary(data)
        assert vec.to_list() == [1.5, 2, 3]
        if NUMPY_AVAILABLE:
            assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])
        assert vec.to_binary() == data
