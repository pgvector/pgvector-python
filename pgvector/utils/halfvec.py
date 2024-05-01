from struct import pack, unpack_from


class HalfVec:
    def __init__(self, value):
        # TODO support np.array
        if not isinstance(value, (list, tuple)):
            raise ValueError('expected list or tuple')

        self.value = value

    def to_list(self):
        return list(self.value)

    def to_db(self):
        return '[' + ','.join([str(float(v)) for v in self.value]) + ']'

    def to_db_binary(self):
        return pack(f'>HH{len(self.value)}e', len(self.value), 0, *self.value)

    def from_db(value):
        return HalfVec([float(v) for v in value[1:-1].split(',')])

    def from_db_binary(value):
        dim, unused = unpack_from('>HH', value)
        return HalfVec(unpack_from(f'>{dim}e', value, 4))

    def __repr__(self):
        return f'HalfVec({self.value})'
