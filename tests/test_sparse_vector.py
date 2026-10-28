from pgvector import SparseVector
import pytest
import struct
from .conftest import numpy as np

try:
    from scipy import sparse
except ImportError:
    sparse = None  # type: ignore


class TestSparseVector:
    def test_list(self) -> None:
        vec = SparseVector([1, 0, 2, 0, 3, 0])
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        if np is not None:
            assert np.array_equal(vec.to_numpy(), [1, 0, 2, 0, 3, 0])
        assert vec.indices() == [0, 2, 4]

    def test_list_empty(self) -> None:
        assert SparseVector([]).to_list() == []

    def test_list_str(self) -> None:
        with pytest.raises(ValueError, match='could not convert string to float'):
            SparseVector([1, 'two', 3])  # type: ignore

    def test_list_dimensions(self) -> None:
        with pytest.raises(ValueError) as error:
            SparseVector([1, 0, 2, 0, 3, 0], 6)  # type: ignore
        assert str(error.value) == 'extra argument'

    def test_ndarray(self) -> None:
        if np is None:
            pytest.skip('NumPy required')

        vec = SparseVector(np.array([1, 0, 2, 0, 3, 0]))
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_dict(self) -> None:
        vec = SparseVector({2: 2, 4: 3, 0: 1, 3: 0}, 6)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_dict_empty(self) -> None:
        assert SparseVector({}, 0).to_list() == []

    # TODO uncomment in 0.6.0
    # def test_dict_negative_index(self) -> None:
    #     with pytest.raises(ValueError, match='sparsevec index out of bounds'):
    #         SparseVector({-1: 1}, 3)

    # TODO uncomment in 0.6.0
    # def test_dict_higher_index(self) -> None:
    #     with pytest.raises(ValueError, match='sparsevec index out of bounds'):
    #         SparseVector({3: 1}, 3)

    # TODO uncomment in 0.6.0
    # def test_dict_negative_dimensions(self) -> None:
    #     with pytest.raises(ValueError, match='sparsevec cannot have negative dimensions'):
    #         SparseVector({}, -1)

    def test_dict_no_dimensions(self) -> None:
        with pytest.raises(ValueError) as error:
            SparseVector({0: 1, 2: 2, 4: 3})  # type: ignore
        assert str(error.value) == 'missing dimensions'

    def test_coo_array(self) -> None:
        if np is None or sparse is None:
            pytest.skip('NumPy and SciPy required')

        arr = sparse.coo_array(([2, 3, 1, 0], ([2, 4, 0, 3],)), shape=(6,))
        vec = SparseVector(arr)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]
        assert isinstance(vec.values()[0], float)

    def test_coo_array_duplicates(self) -> None:
        if np is None or sparse is None:
            pytest.skip('NumPy and SciPy required')

        arr = sparse.coo_array(([1, 2], ([1, 1],)), shape=(3,))
        assert arr.todense().tolist() == [0, 3, 0]
        vec = SparseVector(arr)
        assert vec.to_list() == [0, 3, 0]
        assert vec.indices() == [1]

    def test_coo_array_dimensions(self) -> None:
        if np is None or sparse is None:
            pytest.skip('NumPy and SciPy required')

        with pytest.raises(ValueError) as error:
            SparseVector(sparse.coo_array(np.array([1, 0, 2, 0, 3, 0])), 6)  # type: ignore
        assert str(error.value) == 'extra argument'

    def test_coo_matrix(self) -> None:
        if np is None or sparse is None:
            pytest.skip('NumPy and SciPy required')

        mat = sparse.coo_matrix(np.array([1, 0, 2, 0, 3, 0]))
        vec = SparseVector(mat)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_dok_array(self) -> None:
        if np is None or sparse is None:
            pytest.skip('NumPy and SciPy required')

        arr = sparse.coo_array(np.array([1, 0, 2, 0, 3, 0])).todok()
        vec = SparseVector(arr)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_csr_array(self) -> None:
        if np is None or sparse is None:
            pytest.skip('NumPy and SciPy required')

        arr = sparse.csr_array(np.array([[1, 0, 2, 0, 3, 0]]))
        vec = SparseVector(arr)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_csr_matrix(self) -> None:
        if np is None or sparse is None:
            pytest.skip('NumPy and SciPy required')

        mat = sparse.csr_matrix(np.array([1, 0, 2, 0, 3, 0]))
        vec = SparseVector(mat)
        assert vec.to_list() == [1, 0, 2, 0, 3, 0]
        assert vec.indices() == [0, 2, 4]

    def test_repr(self) -> None:
        assert repr(SparseVector([1, 0, 2, 0, 3, 0])) == 'SparseVector({0: 1.0, 2: 2.0, 4: 3.0}, 6)'
        assert str(SparseVector([1, 0, 2, 0, 3, 0])) == 'SparseVector({0: 1.0, 2: 2.0, 4: 3.0}, 6)'

    def test_equality(self) -> None:
        assert SparseVector([1, 0, 2, 0, 3, 0]) == SparseVector([1, 0, 2, 0, 3, 0])
        assert SparseVector([1, 0, 2, 0, 3, 0]) != SparseVector([1, 0, 2, 0, 3, 1])
        assert SparseVector([1, 0, 2, 0, 3, 0]) == SparseVector({2: 2, 4: 3, 0: 1, 3: 0}, 6)
        assert SparseVector({}, 1) != SparseVector({}, 2)
        assert SparseVector([1, 0, 2, 0, 3, 0]) != 1

    def test_dimensions(self) -> None:
        assert SparseVector([1, 0, 2, 0, 3, 0]).dimensions() == 6

    def test_indices(self) -> None:
        assert SparseVector([1, 0, 2, 0, 3, 0]).indices() == [0, 2, 4]

    def test_values(self) -> None:
        assert SparseVector([1, 0, 2, 0, 3, 0]).values() == [1, 2, 3]

    def test_to_coo(self) -> None:
        if np is None or sparse is None:
            pytest.skip('NumPy and SciPy required')

        assert np.array_equal(SparseVector([1, 0, 2, 0, 3, 0]).to_coo().toarray(), [[1, 0, 2, 0, 3, 0]])

    def test_zero_vector_text(self) -> None:
        vec = SparseVector({}, 3)
        assert vec.to_list() == SparseVector.from_text(vec.to_text()).to_list()

    def test_from_text(self) -> None:
        vec = SparseVector.from_text('{1:1.5,3:2,5:3}/6')
        assert vec.dimensions() == 6
        assert vec.indices() == [0, 2, 4]
        assert vec.values() == [1.5, 2, 3]
        assert vec.to_list() == [1.5, 0, 2, 0, 3, 0]
        if np is not None:
            assert np.array_equal(vec.to_numpy(), [1.5, 0, 2, 0, 3, 0])
        assert vec.to_text() == '{1:1.5,3:2.0,5:3.0}/6'

    # TODO uncomment in 0.6.0
    # def test_from_text_unsorted(self) -> None:
    #     with pytest.raises(ValueError, match='sparsevec indices must be in ascending order'):
    #         SparseVector.from_text('{2:1,1:1}/3')

    # TODO uncomment in 0.6.0
    # def test_from_text_duplicate(self) -> None:
    #     with pytest.raises(ValueError, match='sparsevec indices must be in ascending order'):
    #         SparseVector.from_text('{2:1,2:1}/3')

    def test_from_binary(self) -> None:
        data = struct.pack('>iii3i3f', 6, 3, 0, 0, 2, 4, 1.5, 2, 3)
        vec = SparseVector.from_binary(data)
        assert vec.dimensions() == 6
        assert vec.indices() == [0, 2, 4]
        assert vec.values() == [1.5, 2, 3]
        assert vec.to_list() == [1.5, 0, 2, 0, 3, 0]
        if np is not None:
            assert np.array_equal(vec.to_numpy(), [1.5, 0, 2, 0, 3, 0])
        assert vec.to_binary() == data

    def test_from_binary_zero_length(self) -> None:
        # TODO change to ValueError('invalid length') in 0.6.0
        with pytest.raises(struct.error, match='unpack_from requires a buffer of at least 12 bytes'):
            SparseVector.from_binary(b'')

    def test_from_binary_invalid_nnz(self) -> None:
        with pytest.raises(ValueError, match='invalid length'):
            SparseVector.from_binary(struct.pack('>iii', 1, 1, 0))

    def test_from_binary_negative_nnz(self) -> None:
        with pytest.raises(ValueError, match='invalid length'):
            SparseVector.from_binary(struct.pack('>iii', 0, -1, 0))

    def test_from_binary_non_zero_unused(self) -> None:
        with pytest.raises(ValueError, match='expected unused to be 0'):
            SparseVector.from_binary(struct.pack('>iii', 0, 0, 1))
