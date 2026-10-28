from pgvector import Vector
import pytest
import struct
from .conftest import numpy as np


class TestVector:
    def test_init_list(self) -> None:
        arr = [1.0, 2.0, 3.0]
        assert Vector(arr).to_list() == arr
        assert Vector(arr).to_list() is not arr

    def test_init_list_empty(self) -> None:
        assert Vector([]).to_list() == []

    def test_init_list_str(self) -> None:
        with pytest.raises(ValueError) as error:
            Vector([1, 'two', 3])  # type: ignore
        assert str(error.value) == 'expected list[float]'

    def test_init_list_list(self) -> None:
        with pytest.raises(ValueError) as error:
            Vector([[1, 2], [3, 4]])  # type: ignore
        assert str(error.value) == 'expected list[float]'

    def test_init_ndarray(self) -> None:
        if np is None:
            pytest.skip('NumPy required')

        arr = np.array([1, 2, 3], dtype=np.float32)
        assert Vector(arr).to_list() == [1, 2, 3]
        assert Vector(arr).to_numpy() is not arr
        assert Vector(arr).to_numpy().dtype == np.float32

    def test_init_ndarray_non_contiguous(self) -> None:
        if np is None:
            pytest.skip('NumPy required')

        arr = np.array([1, 2, 3], dtype=np.float32)
        assert Vector(np.flip(arr)).to_list() == [3, 2, 1]
        assert Vector(np.flip(arr)).to_binary() == Vector([3, 2, 1]).to_binary()

    def test_init_ndarray_big_endian(self) -> None:
        if np is None:
            pytest.skip('NumPy required')

        arr = np.array([1, 2, 3], dtype=np.float32)
        assert Vector(arr.astype('>f4')).to_list() == [1, 2, 3]
        assert Vector(arr.astype('>f4')).to_binary() == Vector([1, 2, 3]).to_binary()

    def test_init_ndarray_str(self) -> None:
        if np is None:
            pytest.skip('NumPy required')

        with pytest.raises(ValueError) as error:
            Vector(np.array(['one', 'two', 'three']))
        assert 'could not convert string to float' in str(error.value)

    def test_init_int(self) -> None:
        with pytest.raises(ValueError) as error:
            Vector(1)  # type: ignore
        assert str(error.value) == 'expected list or ndarray'

    def test_repr(self) -> None:
        assert repr(Vector([1, 2, 3])) == 'Vector([1.0, 2.0, 3.0])'
        assert str(Vector([1, 2, 3])) == 'Vector([1.0, 2.0, 3.0])'

    def test_eq(self) -> None:
        assert Vector([1, 2, 3]) == Vector([1, 2, 3])
        assert Vector([1, 2, 3]) != Vector([1, 2, 4])
        assert Vector([1, 2, 3]) != 1
        assert Vector([0.0]) == Vector([-0.0])

    def test_dimensions(self) -> None:
        assert Vector([1, 2, 3]).dimensions() == 3

    def test_to_list(self) -> None:
        assert Vector([1.5, 2, 3]).to_list() == [1.5, 2, 3]

    def test_to_numpy(self) -> None:
        if np is None:
            pytest.skip('NumPy required')

        arr = Vector([1.5, 2, 3]).to_numpy()
        assert np.array_equal(arr, np.array([1.5, 2, 3]))
        assert arr.dtype == np.float32

    def test_to_text(self) -> None:
        assert Vector([1.5, 2, 3]).to_text() == '[1.5,2.0,3.0]'

    def test_to_binary(self) -> None:
        assert Vector([1.5, 2, 3]).to_binary() == struct.pack('>hh3f', 3, 0, 1.5, 2, 3)

    def test_from_text(self) -> None:
        assert Vector.from_text('[1.5,2,3]') == Vector([1.5, 2, 3])

    def test_from_text_zero_length(self) -> None:
        with pytest.raises(ValueError, match='could not convert string to float'):
            Vector.from_text('')

    def test_from_text_zero_dimensions(self) -> None:
        assert Vector.from_text('[]') == Vector([])

    def test_from_text_open_bracket(self) -> None:
        # TODO raise ValueError in 0.6.0
        assert Vector.from_text('{1]')

    def test_from_text_close_bracket(self) -> None:
        # TODO raise ValueError in 0.6.0
        assert Vector.from_text('[1}')

    def test_from_text_non_number(self) -> None:
        with pytest.raises(ValueError, match='could not convert string to float'):
            assert Vector.from_text('[a]')

    def test_from_binary(self) -> None:
        assert Vector.from_binary(struct.pack('>hh3f', 3, 0, 1.5, 2, 3)) == Vector([1.5, 2, 3])

    def test_from_binary_zero_length(self) -> None:
        # TODO change to ValueError('invalid length') in 0.6.0
        with pytest.raises(struct.error, match='unpack_from requires a buffer of at least 4 bytes'):
            Vector.from_binary(b'')

    def test_from_binary_missing_data(self) -> None:
        with pytest.raises(ValueError, match='invalid length'):
            Vector.from_binary(struct.pack('>hhf', 2, 0, 1))

    def test_from_binary_extra_data(self) -> None:
        with pytest.raises(ValueError, match='invalid length'):
            Vector.from_binary(struct.pack('>hhfff', 2, 0, 1, 2, 3))

    def test_from_binary_negative_dimensions(self) -> None:
        with pytest.raises(ValueError, match='invalid length'):
            Vector.from_binary(struct.pack('>hh', -1, 0))

    def test_from_binary_non_zero_unused(self) -> None:
        with pytest.raises(ValueError, match='expected unused to be 0'):
            Vector.from_binary(struct.pack('>hh', 0, 1))
