import numpy as np


def from_db(value):
    # could be ndarray if already cast by lower-level driver
    if value is None or isinstance(value, np.ndarray):
        return value

    return np.array(value[1:-1].split(','), dtype=np.float32)


def to_db(value, dim=None):
    if value is None:
        return value

    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError('expected ndim to be 1')

        if not np.issubdtype(value.dtype, np.integer) and not np.issubdtype(value.dtype, np.floating):
            raise ValueError('dtype must be numeric')

        value = value.tolist()

    if dim is not None and len(value) != dim:
        raise ValueError('expected %d dimensions, not %d' % (dim, len(value)))

    return '[' + ','.join([str(float(v)) for v in value]) + ']'
