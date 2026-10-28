from pgvector import HalfVector
import pytest
import struct
from .conftest import numpy as np


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

    def test_ndarray(self) -> None:
        if np is None:
            pytest.skip('NumPy required')

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
        assert HalfVector([1, 2, 3]) != 1
        assert HalfVector([0.0]) == HalfVector([-0.0])

    def test_dimensions(self) -> None:
        assert HalfVector([1, 2, 3]).dimensions() == 3

    def test_from_text(self) -> None:
        vec = HalfVector.from_text('[1.5,2,3]')
        assert vec.to_list() == [1.5, 2, 3]
        if np is not None:
            assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])
        assert vec.to_text() == '[1.5,2.0,3.0]'

    def test_from_text_zero_dimensions(self) -> None:
        assert HalfVector.from_text('[]') == HalfVector([])

    def test_from_text_open_bracket(self) -> None:
        # TODO raise ValueError in 0.6.0
        assert HalfVector.from_text('{1]')

    def test_from_text_close_bracket(self) -> None:
        # TODO raise ValueError in 0.6.0
        assert HalfVector.from_text('[1}')

    def test_from_binary(self) -> None:
        data = struct.pack('>HH3e', 3, 0, 1.5, 2, 3)
        vec = HalfVector.from_binary(data)
        assert vec.to_list() == [1.5, 2, 3]
        if np is not None:
            assert np.array_equal(vec.to_numpy(), [1.5, 2, 3])
        assert vec.to_binary() == data

    def test_from_binary_zero_length(self) -> None:
        # TODO change to ValueError('invalid length') in 0.6.0
        with pytest.raises(struct.error, match='unpack_from requires a buffer of at least 4 bytes'):
            HalfVector.from_binary(b'')

    def test_from_binary_low_dimensions(self) -> None:
        with pytest.raises(ValueError, match='invalid length'):
            HalfVector.from_binary(struct.pack('>hhee', 1, 0, 1, 2))

    def test_from_binary_high_dimensions(self) -> None:
        with pytest.raises(ValueError, match='invalid length'):
            HalfVector.from_binary(struct.pack('>hhee', 3, 0, 1, 2))

    def test_from_binary_negative_dimensions(self) -> None:
        with pytest.raises(ValueError, match='invalid length'):
            HalfVector.from_binary(struct.pack('>hh', -1, 0))

    def test_from_binary_non_zero_unused(self) -> None:
        with pytest.raises(ValueError, match='expected unused to be 0'):
            HalfVector.from_binary(struct.pack('>hh', 0, 1))
