from django.contrib.postgres.operations import CreateExtension
from .functions import L2Distance, MaxInnerProduct, CosineDistance, L1Distance
from .indexes import IvfflatIndex, HnswIndex
from .vector import VectorField

__all__ = ['VectorExtension', 'VectorField', 'IvfflatIndex', 'HnswIndex', 'L2Distance', 'MaxInnerProduct', 'CosineDistance', 'L1Distance']


class VectorExtension(CreateExtension):
    def __init__(self):
        self.name = 'vector'
