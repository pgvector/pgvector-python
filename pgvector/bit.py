from __future__ import annotations
from struct import pack, unpack_from
from typing import TYPE_CHECKING
from ._utils import is_ndarray

if TYPE_CHECKING:
    import numpy as np


class Bit:
    _length: int
    _data: bytes

    def __init__(
        self,
        value: bytes | str | list[bool] | np.ndarray[tuple[int, ...], np.dtype[np.bool | np.uint8]],
        /
    ) -> None:
        if isinstance(value, bytes):
            self._length = 8 * len(value)
            self._data = value
        elif isinstance(value, (list, str)):
            if isinstance(value, list):
                bits = {True: '1', False: '0'}
                try:
                    value = ''.join([bits[v] for v in value])
                except (KeyError, TypeError):
                    raise ValueError('expected list[bool]')

            length = len(value)
            if length % 8 != 0:
                value += '0' * (8 - (length % 8))

            self._length = length
            try:
                self._data = int(value, 2).to_bytes(len(value) // 8, byteorder='big')
            except ValueError:
                raise ValueError('expected bit string')
        elif is_ndarray(value):
            import numpy as np

            if value.dtype != np.bool:
                # skip error for result of np.unpackbits
                if value.dtype != np.uint8 or np.any(value > 1):
                    raise ValueError('expected elements to be boolean')
                value = value.astype(bool)

            if value.ndim != 1:
                raise ValueError('expected ndim to be 1')

            self._length = len(value)
            self._data = np.packbits(value).tobytes()  # type: ignore
        else:
            raise ValueError('expected bytes, str, list, or ndarray')

    def __repr__(self) -> str:
        return f'Bit({self.to_text()})'

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._length == other._length and self._data == other._data

    def to_list(self) -> list[bool]:
        # TODO improve
        return [v != '0' for v in self.to_text()]

    def to_numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.bool]]:
        import numpy as np

        return np.unpackbits(np.frombuffer(self._data, dtype=np.uint8), count=self._length).astype(bool)

    def to_text(self) -> str:
        return ''.join(format(v, '08b') for v in self._data)[:self._length]

    def to_binary(self) -> bytes:
        return pack('>i', self._length) + self._data

    @classmethod
    def from_text(cls, value: str, /) -> Bit:
        # cast to ensure always uses str constructor
        return cls(str(value))

    @classmethod
    def from_binary(cls, value: bytes | bytearray | memoryview, /) -> Bit:
        length, = unpack_from('>i', value)
        data = memoryview(value)[4:].tobytes()

        if len(data) != (length + 7) // 8:
            raise ValueError('invalid length')

        bit = cls.__new__(cls)
        bit._length = length
        bit._data = data
        return bit
