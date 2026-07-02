from pgvector import HalfVector
import pytest
from struct import pack

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestHalfVector:
    def test_list(self) -> None:
        arr = [1.0, 2.0, 3.0]
        assert HalfVector(arr).to_list() == arr
        assert HalfVector(arr).to_list() is not arr

    def test_list_empty(self) -> None:
        assert HalfVector([]).to_list() == []

    def test_list_str(self) -> None:
        with pytest.raises(ValueError) as error:
            HalfVector([1, 'two', 3])  # type: ignore
        assert str(error.value) == 'expected list[float]'

    def test_list_list(self) -> None:
        with pytest.raises(ValueError) as error:
            HalfVector([[1, 2], [3, 4]])  # type: ignore
        assert str(error.value) == 'expected list[float]'

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason='NumPy required')
    def test_ndarray(self) -> None:
        arr = np.array([1, 2, 3], dtype=np.float16)
        assert HalfVector(arr).to_list() == [1, 2, 3]
        assert HalfVector(arr).to_numpy() is not arr
        assert HalfVector(arr).to_numpy().dtype == np.float16
        # non-contiguous
        assert HalfVector(np.flip(arr)).to_list() == [3, 2, 1]
        assert HalfVector(np.flip(arr)).to_binary() == HalfVector([3, 2, 1]).to_binary()
        # big endian
        assert HalfVector(arr.astype('>f2')).to_list() == [1, 2, 3]
        assert HalfVector(arr.astype('>f2')).to_binary() == HalfVector([1, 2, 3]).to_binary()

        with pytest.raises(ValueError) as error:
            HalfVector(np.array(['one', 'two', 'three']))
        assert 'could not convert string to float' in str(error.value)

    def test_int(self) -> None:
        with pytest.raises(ValueError) as error:
            HalfVector(1)  # type: ignore
        assert str(error.value) == 'expected list or ndarray'

    def test_repr(self) -> None:
        assert repr(HalfVector([1, 2, 3])) == 'HalfVector([1.0, 2.0, 3.0])'
        assert str(HalfVector([1, 2, 3])) == 'HalfVector([1.0, 2.0, 3.0])'

    def test_equality(self) -> None:
        assert HalfVector([1, 2, 3]) == HalfVector([1, 2, 3])
        assert HalfVector([1, 2, 3]) != HalfVector([1, 2, 4])

    def test_dimensions(self) -> None:
        assert HalfVector([1, 2, 3]).dimensions() == 3

    def test_from_text(self) -> None:
        vec = HalfVector.from_text('[1.5,2,3]')
        assert vec.to_list() == [1.5, 2, 3]
        if NUMPY_AVAILABLE:
            assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])

    def test_from_binary(self) -> None:
        data = pack('>HH3e', 3, 0, 1.5, 2, 3)
        vec = HalfVector.from_binary(data)
        assert vec.to_list() == [1.5, 2, 3]
        if NUMPY_AVAILABLE:
            assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])
        assert vec.to_binary() == data
