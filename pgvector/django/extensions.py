from django.contrib.postgres.operations import CreateExtension


class VectorExtension(CreateExtension):
    def __init__(self):
        self.name = 'vector'
