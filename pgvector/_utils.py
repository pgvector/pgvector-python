import sys


def is_ndarray(value: object) -> bool:
    if (numpy := sys.modules.get('numpy')):
        return isinstance(value, numpy.ndarray)
    return False


def is_sparse_array(value: object) -> bool:
    if (sparse := sys.modules.get('scipy.sparse')):
        return isinstance(value, (sparse.sparray, sparse.spmatrix))
    return False
