from pgvector import Bit
import pytest
import random
import struct
from .conftest import numpy as np


class TestBit:
    def test_init_list(self) -> None:
        assert Bit([True, False, True]).to_list() == [True, False, True]

    def test_init_list_empty(self) -> None:
        assert Bit([]).to_list() == []

    def test_init_list_none(self) -> None:
        with pytest.raises(ValueError) as error:
            Bit([True, None, True])  # type: ignore
        assert str(error.value) == 'expected list[bool]'

    def test_init_list_int(self) -> None:
        with pytest.raises(ValueError) as error:
            Bit([254, 7, 0])  # type: ignore
        assert str(error.value) == 'expected list[bool]'

    def test_init_list_list(self) -> None:
        with pytest.raises(ValueError) as error:
            Bit([[True, False], [True, False]])  # type: ignore
        assert str(error.value) == 'expected list[bool]'

    def test_init_str(self) -> None:
        assert Bit('101').to_list() == [True, False, True]

    def test_init_str_empty(self) -> None:
        assert Bit('').to_list() == []

    def test_init_str_two(self) -> None:
        with pytest.raises(ValueError) as error:
            Bit('201')
        assert str(error.value) == 'expected bit string'

    def test_init_bytes(self) -> None:
        assert Bit(b'\xff\x00\xf0').to_text() == '111111110000000011110000'
        assert Bit(b'\xfe\x07\x00').to_text() == '111111100000011100000000'

    def test_init_bytes_empty(self) -> None:
        assert Bit(b'').to_text() == ''

    def test_init_ndarray(self) -> None:
        if np is None:
            pytest.skip('NumPy required')

        arr = np.array([True, False, True])
        assert Bit(arr).to_list() == [True, False, True]
        assert np.array_equal(Bit(arr).to_numpy(), arr)

    def test_init_ndarray_unpackbits(self) -> None:
        if np is None:
            pytest.skip('NumPy required')

        arr = np.unpackbits(np.array([254, 7, 0], dtype=np.uint8))
        assert Bit(arr).to_text() == '111111100000011100000000'

    def test_init_ndarray_uint8(self) -> None:
        if np is None:
            pytest.skip('NumPy required')

        arr = np.array([254, 7, 0], dtype=np.uint8)
        with pytest.raises(ValueError) as error:
            Bit(arr)
        assert str(error.value) == 'expected elements to be boolean'

    def test_init_ndarray_uint16(self) -> None:
        if np is None:
            pytest.skip('NumPy required')

        arr = np.array([254, 7, 0], dtype=np.uint16)
        with pytest.raises(ValueError) as error:
            Bit(arr)  # type: ignore
        assert str(error.value) == 'expected elements to be boolean'

    def test_init_bool(self) -> None:
        with pytest.raises(ValueError) as error:
            Bit(True)  # type: ignore
        assert str(error.value) == 'expected bytes, str, list, or ndarray'

    def test_init_random(self) -> None:
        value = ''.join(random.choices(['0', '1'], k=random.randint(1024, 2048)))
        assert Bit(value).to_text() == value

    def test_repr(self) -> None:
        assert repr(Bit([True, False, True])) == 'Bit(101)'
        assert str(Bit([True, False, True])) == 'Bit(101)'

    def test_eq(self) -> None:
        assert Bit([True, False, True]) == Bit([True, False, True])
        assert Bit([True, False, True]) != Bit([True, False, False])
        assert Bit([True, False, True]) != 1

    def test_from_text(self) -> None:
        assert Bit.from_text('101') == Bit('101')

    def test_from_text_zero_length(self) -> None:
        assert Bit.from_text('') == Bit([])

    def test_from_text_invalid_character(self) -> None:
        with pytest.raises(ValueError, match='expected bit string'):
            Bit.from_text('2')

    def test_from_binary(self) -> None:
        assert Bit.from_binary(struct.pack('>iB', 3, 5 << 5)) == Bit('101')

    def test_from_binary_zero_length(self) -> None:
        # TODO change to ValueError('invalid length') in 0.6.0
        with pytest.raises(struct.error, match='unpack_from requires a buffer of at least 4 bytes'):
            Bit.from_binary(b'')

    def test_from_binary_low_length(self) -> None:
        with pytest.raises(ValueError, match='invalid length'):
            Bit.from_binary(struct.pack('>iBB', 8, 0, 0))

    def test_from_binary_high_length(self) -> None:
        with pytest.raises(ValueError, match='invalid length'):
            Bit.from_binary(struct.pack('>iBB', 17, 0, 0))

    def test_from_binary_negative_length(self) -> None:
        with pytest.raises(ValueError, match='invalid length'):
            Bit.from_binary(struct.pack('>i', -1))
