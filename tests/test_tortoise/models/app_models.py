from tortoise.models import Model
from tortoise import fields

from pgvector.tortoise import VectorField


class Item(Model):
    id = fields.IntField(primary_key=True)
    embedding = VectorField(dimensions=3)

    def __str__(self):
        return self.id

    class Meta:
        table = "tortoise_items"
