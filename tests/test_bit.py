
import numpy as np
import pytest
from pgvector.utils import Bit


class TestBit:
    def test_to_db(self):
        assert Bit([True, False, True]).to_db() == '101'
        assert Bit([True, True, True]).to_db() == '111'
        assert Bit([False, False, False]).to_db() == '000'


    def test_to_db_binary(self):
        bin_value = b'\x00\x00\x00\x03\xa0'
        assert Bit([True, False, True]).to_db_binary() == bin_value