import sys
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np

    ndarray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.floating]]


def is_ndarray(value: object) -> bool:
    if (numpy := sys.modules.get('numpy')):
        return isinstance(value, numpy.ndarray)
    return False


def is_sparse_array(value: object) -> bool:
    if (sparse := sys.modules.get('scipy.sparse')):
        return isinstance(value, (sparse.sparray, sparse.spmatrix))
    return False
