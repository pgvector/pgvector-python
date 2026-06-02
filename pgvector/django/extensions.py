from django import VERSION
from django.contrib.postgres.operations import CreateExtension


class VectorExtension(CreateExtension):
    def __init__(self, hints=None):
        if VERSION[0] >= 6:
            super().__init__('vector', hints=hints)
        else:
            self.name = 'vector'
