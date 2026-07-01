from pgvector import Bit
import pytest
import random

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestBit:
    def test_list(self) -> None:
        assert Bit([True, False, True]).to_list() == [True, False, True]

    def test_list_none(self) -> None:
        with pytest.raises(ValueError) as error:
            Bit([True, None, True])  # type: ignore
        assert str(error.value) == 'expected list[bool]'

    def test_list_int(self) -> None:
        with pytest.raises(ValueError) as error:
            Bit([254, 7, 0])  # type: ignore
        assert str(error.value) == 'expected list[bool]'

    def test_list_list(self) -> None:
        with pytest.raises(ValueError) as error:
            Bit([[True, False], [True, False]])  # type: ignore
        assert str(error.value) == 'expected list[bool]'

    def test_str(self) -> None:
        assert Bit('101').to_list() == [True, False, True]

    def test_str_two(self) -> None:
        with pytest.raises(ValueError) as error:
            Bit('201')
        assert str(error.value) == 'expected bit string'

    def test_bytes(self) -> None:
        assert Bit(b'\xff\x00\xf0').to_text() == '111111110000000011110000'
        assert Bit(b'\xfe\x07\x00').to_text() == '111111100000011100000000'

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason='NumPy required')
    def test_ndarray(self) -> None:
        arr = np.array([True, False, True])
        assert Bit(arr).to_list() == [True, False, True]
        assert np.array_equal(Bit(arr).to_numpy(), arr)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason='NumPy required')
    def test_ndarray_unpackbits(self) -> None:
        arr = np.unpackbits(np.array([254, 7, 0], dtype=np.uint8))
        assert Bit(arr).to_text() == '111111100000011100000000'

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason='NumPy required')
    def test_ndarray_uint8(self) -> None:
        arr = np.array([254, 7, 0], dtype=np.uint8)
        with pytest.raises(ValueError) as error:
            Bit(arr)
        assert str(error.value) == 'expected elements to be boolean'

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason='NumPy required')
    def test_ndarray_uint16(self) -> None:
        arr = np.array([254, 7, 0], dtype=np.uint16)
        with pytest.raises(ValueError) as error:
            Bit(arr)  # type: ignore
        assert str(error.value) == 'expected elements to be boolean'

    def test_bool(self) -> None:
        with pytest.raises(ValueError) as error:
            Bit(True)  # type: ignore
        assert str(error.value) == 'expected bytes, str, list, or ndarray'

    def test_random(self) -> None:
        value = ''.join(random.choices(['0', '1'], k=random.randint(1024, 2048)))
        assert Bit(value).to_text() == value

    def test_repr(self) -> None:
        assert repr(Bit([True, False, True])) == 'Bit(101)'
        assert str(Bit([True, False, True])) == 'Bit(101)'

    def test_equality(self) -> None:
        assert Bit([True, False, True]) == Bit([True, False, True])
        assert Bit([True, False, True]) != Bit([True, False, False])
