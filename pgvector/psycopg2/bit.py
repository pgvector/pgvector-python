from psycopg2.extensions import adapt, register_adapter
from ..utils import Bit


class BitAdapter:
    def __init__(self, value):
        self._value = value

    def getquoted(self):
        return adapt(Bit.to_db(self._value)).getquoted()


def register_bit_info():
    register_adapter(Bit, BitAdapter)
