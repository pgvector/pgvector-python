from psycopg2.extensions import adapt, register_adapter
from typing import Any
from .. import Bit


class BitAdapter:
    def __init__(self, value: Bit) -> None:
        self._value = value

    def getquoted(self) -> Any:
        return adapt(self._value.to_text()).getquoted()


def register_bit_info() -> None:
    register_adapter(Bit, BitAdapter)
