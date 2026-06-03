from pgvector import Bit
import pytest

try:
    import numpy as np
except ImportError:
    np = None


class TestBit:
    def test_list(self):
        assert Bit([True, False, True]).to_list() == [True, False, True]

    def test_list_none(self):
        with pytest.raises(ValueError) as error:
            Bit([True, None, True])  # ty: ignore[invalid-argument-type]
        assert str(error.value) == 'expected list[bool]'

    def test_list_int(self):
        with pytest.raises(ValueError) as error:
            Bit([254, 7, 0])  # ty: ignore[invalid-argument-type]
        assert str(error.value) == 'expected list[bool]'

    def test_str(self):
        assert Bit('101').to_list() == [True, False, True]

    def test_str_two(self):
        with pytest.raises(ValueError) as error:
            Bit('201')
        assert str(error.value) == 'expected bit string'

    def test_bytes(self):
        assert Bit(b'\xff\x00\xf0').to_text() == '111111110000000011110000'
        assert Bit(b'\xfe\x07\x00').to_text() == '111111100000011100000000'

    @pytest.mark.skipif(np is None, reason='NumPy required')
    def test_ndarray(self):
        arr = np.array([True, False, True])
        assert Bit(arr).to_list() == [True, False, True]
        assert np.array_equal(Bit(arr).to_numpy(), arr)

    @pytest.mark.skipif(np is None, reason='NumPy required')
    def test_ndarray_unpackbits(self):
        arr = np.unpackbits(np.array([254, 7, 0], dtype=np.uint8))
        assert Bit(arr).to_text() == '111111100000011100000000'

    @pytest.mark.skipif(np is None, reason='NumPy required')
    def test_ndarray_uint8(self):
        arr = np.array([254, 7, 0], dtype=np.uint8)
        with pytest.warns(UserWarning, match='expected elements to be boolean'):
            assert Bit(arr).to_text() == '110'

    @pytest.mark.skipif(np is None, reason='NumPy required')
    def test_ndarray_uint16(self):
        arr = np.array([254, 7, 0], dtype=np.uint16)
        with pytest.warns(UserWarning, match='expected elements to be boolean'):
            assert Bit(arr).to_text() == '110'

    def test_ndim_two(self):
        with pytest.raises(ValueError) as error:
            Bit([[True, False], [True, False]])  # ty: ignore[invalid-argument-type]
        assert str(error.value) == 'expected list[bool]'

    def test_ndim_zero(self):
        with pytest.raises(ValueError) as error:
            Bit(True)  # ty: ignore[invalid-argument-type]
        assert str(error.value) == 'expected bytes, str, list, or ndarray'

    def test_repr(self):
        assert repr(Bit([True, False, True])) == 'Bit(101)'
        assert str(Bit([True, False, True])) == 'Bit(101)'

    def test_equality(self):
        assert Bit([True, False, True]) == Bit([True, False, True])
        assert Bit([True, False, True]) != Bit([True, False, False])
