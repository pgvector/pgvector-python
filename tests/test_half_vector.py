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
        assert HalfVector([1, 2, 3]).to_list() == [1, 2, 3]

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
        arr = np.array([1, 2, 3])
        assert HalfVector(arr).to_list() == [1, 2, 3]
        assert HalfVector(arr).to_numpy() is not arr

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

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason='NumPy required')
    def test_to_numpy_readonly(self) -> None:
        arr = HalfVector([1, 2, 3]).to_numpy()
        with pytest.raises(ValueError) as error:
            arr[0] = 4
        assert str(error.value) == 'assignment destination is read-only'

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
