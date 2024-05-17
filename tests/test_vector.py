import numpy as np
import pytest
from pgvector.utils import Vector 


class TestVector:
    def test_from_db(self):
        assert Vector.from_db(None) is None
        assert np.array_equal(Vector.from_db('[1,2,3]'), np.array([1, 2, 3], dtype=np.float32))
    
    def test_from_db_binary(self):
        value = b'\x00\x03\x00\x00?\x80\x00\x00@\x00\x00\x00@@\x00\x00'
        assert Vector.from_db_binary(None) is None
        assert np.array_equal(Vector.from_db_binary(value), np.array([1, 2, 3], dtype=np.float32))


    def test_to_db(self):
        assert Vector.to_db(None) is None
        assert Vector.to_db(np.array([1, 2, 3], dtype=np.float32)) == '[1.0,2.0,3.0]'
        with pytest.raises(ValueError, match='expected ndim to be 1'):
            Vector.to_db(np.array([[1, 2], [3, 4]], dtype=np.float32))
        with pytest.raises(ValueError, match='dtype must be numeric'):  
            Vector.to_db(np.array([True, False, True], dtype=bool))
        with pytest.raises(ValueError, match='expected 4 dimensions, not 3'):
            Vector.to_db([1, 2, 3], dim=4)
        assert Vector.to_db([1, 2, 3], dim=3) == '[1.0,2.0,3.0]'    


    def test_to_db_binary(self):
        value = [1, 2, 3]
        binary_data = Vector.to_db_binary(value)
        unpacked_values = np.frombuffer(binary_data, dtype='>f', count=3, offset=4).astype(dtype=np.float32)
        np.testing.assert_array_equal(unpacked_values, np.array(value, dtype=np.float32), "The unpacked values should match the original list converted to float32")



    