import numpy as np
from pgvector.utils import Bit
import pytest


class TestBit:
    def test_list(self):
        assert Bit([True, False, True]).to_list() == [True, False, True]

    def test_tuple(self):
        assert Bit((True, False, True)).to_list() == [True, False, True]

    def test_str(self):
        assert str(Bit('101')) == '101'

    def test_ndarray_same_object(self):
        arr = np.array([True, False, True], dtype=bool)
        assert Bit(arr).to_list() == [True, False, True]
        assert Bit(arr).to_numpy() is arr
