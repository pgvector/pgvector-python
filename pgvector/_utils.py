import sys
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np

    ndarray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.floating]]
else:
    # any value works since not type checking
    # TODO use Never when Python 3.10 no longer supported
    ndarray = None


def is_ndarray(value: object, /) -> bool:
    if (numpy := sys.modules.get('numpy')):
        return isinstance(value, numpy.ndarray)
    return False


def is_sparse_array(value: object, /) -> bool:
    if (sparse := sys.modules.get('scipy.sparse')):
        return isinstance(value, (sparse.sparray, sparse.spmatrix))
    return False
