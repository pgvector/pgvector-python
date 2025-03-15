import numpy as np
from pgvector import Bit
import pytest


class TestBit:
    def test_list(self):
        assert Bit([True, False, True]).to_list() == [True, False, True]

    def test_list_none(self):
        with pytest.warns(UserWarning, match='expected elements to be boolean'):
            assert Bit([True, None, True]).to_text() == '101'

    def test_list_int(self):
        with pytest.warns(UserWarning, match='expected elements to be boolean'):
            assert Bit([254, 7, 0]).to_text() == '110'

    def test_tuple(self):
        assert Bit((True, False, True)).to_list() == [True, False, True]

    def test_str(self):
        assert Bit('101').to_list() == [True, False, True]

    def test_bytes(self):
        assert Bit(b'\xff\x00').to_list() == [True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False]
        assert Bit(b'\xfe\x07').to_list() == [True, True, True, True, True, True, True, False, False, False, False, False, False, True, True, True]

    def test_ndarray(self):
        arr = np.array([True, False, True])
        assert Bit(arr).to_list() == [True, False, True]
        assert np.array_equal(Bit(arr).to_numpy(), arr)

    def test_ndarray_uint8(self):
        arr = np.array([254, 7, 0], dtype=np.uint8)
        with pytest.warns(UserWarning, match='expected elements to be boolean'):
            assert Bit(arr).to_text() == '110'

    def test_ndarray_uint16(self):
        arr = np.array([254, 7, 0], dtype=np.uint16)
        with pytest.warns(UserWarning, match='expected elements to be boolean'):
            assert Bit(arr).to_text() == '110'

    def test_ndim_two(self):
        with pytest.raises(ValueError) as error:
            Bit([[True, False], [True, False]])
        assert str(error.value) == 'expected ndim to be 1'

    def test_ndim_zero(self):
        with pytest.raises(ValueError) as error:
            Bit(True)
        assert str(error.value) == 'expected ndim to be 1'

    def test_repr(self):
        assert repr(Bit([True, False, True])) == 'Bit(101)'
        assert str(Bit([True, False, True])) == 'Bit(101)'

    def test_equality(self):
        assert Bit([True, False, True]) == Bit([True, False, True])
        assert Bit([True, False, True]) != Bit([True, False, False])
