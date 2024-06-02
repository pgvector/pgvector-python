import numpy as np
from pgvector.utils import Bit
import pytest


class TestBit:
    def test_list(self):
        assert str(Bit([True, False, True])) == '101'

    def test_str(self):
        assert str(Bit('101')) == '101'
