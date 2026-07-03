from django import VERSION
from django.contrib.postgres.operations import CreateExtension
from typing import Any


class VectorExtension(CreateExtension):
    def __init__(self, hints: Any = None) -> None:
        if VERSION[0] >= 6:
            super().__init__('vector', hints=hints)  # type: ignore
        else:
            self.name = 'vector'
