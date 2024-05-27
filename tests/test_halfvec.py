import numpy as np
import pytest
from pgvector.utils import HalfVector


class TestHalfVector:
    def test_to_db(self):
        assert HalfVector.to_db(None) is None
        assert HalfVector.to_db([1, 2, 3]) == '[1.0,2.0,3.0]'
        with pytest.raises(ValueError, match='expected 4 dimensions, not 3'):
            HalfVector.to_db([1, 2, 3], dim=4)
        assert HalfVector.to_db([1, 2, 3], dim=3) == '[1.0,2.0,3.0]'

    def test_to_db_binary(self):
        value = [1, 2, 3]
        binary_data = HalfVector.to_db_binary(value)
        assert binary_data == b'\x00\x03\x00\x00<\x00@\x00B\x00'

    def test_from_db(self):
        assert HalfVector.from_db(None) is None
        assert HalfVector.from_db('[1,2,3]').value == [1.0, 2.0, 3.0]
    
    def test_from_db_binary(self):
        value = b'\x00\x03\x00\x00<\x00@\x00B\x00'
        assert HalfVector.from_db_binary(None) is None
        assert HalfVector.from_db_binary(value).value == [1.0, 2.0, 3.0]