import numpy as np
from pgvector.utils import Bit
import pytest


class TestBit:
    def test_list(self):
        assert Bit([True, False, True]).to_list() == [True, False, True]

    def test_tuple(self):
        assert Bit((True, False, True)).to_list() == [True, False, True]

    def test_str(self):
        assert Bit('101').to_list() == [True, False, True]

    def test_ndarray_same_object(self):
        arr = np.array([True, False, True])
        assert Bit(arr).to_list() == [True, False, True]
        assert Bit(arr).to_numpy() is arr

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
