from math import sqrt
import numpy as np
from peewee import Model, PostgresqlDatabase
from pgvector.peewee import VectorField

db = PostgresqlDatabase('pgvector_python_test')


class BaseModel(Model):
    class Meta:
        database = db


class Item(BaseModel):
    embedding = VectorField(dimensions=3)


Item.add_index('embedding vector_l2_ops', using='hnsw')

db.connect()
db.execute_sql('CREATE EXTENSION IF NOT EXISTS vector')
db.drop_tables([Item])
db.create_tables([Item])


def create_items():
    vectors = [
        [1, 1, 1],
        [2, 2, 2],
        [1, 1, 2]
    ]
    for i, v in enumerate(vectors):
        Item.create(id=i + 1, embedding=v)


class TestPeewee:
    def setup_method(self, test_method):
        Item.truncate_table()

    def test_works(self):
        Item.create(id=1, embedding=[1, 2, 3])
        item = Item.get_by_id(1)
        assert np.array_equal(item.embedding, np.array([1, 2, 3]))
        assert item.embedding.dtype == np.float32

    def test_l2_distance(self):
        create_items()
        distance = Item.embedding.l2_distance([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]

    def test_max_inner_product(self):
        create_items()
        distance = Item.embedding.max_inner_product([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [-6, -4, -3]

    def test_cosine_distance(self):
        create_items()
        distance = Item.embedding.cosine_distance([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [1, 2, 3]
        assert [v.distance for v in items] == [0, 0, 0.05719095841793653]

    def test_where(self):
        create_items()
        items = Item.select().where(Item.embedding.l2_distance([1, 1, 1]) < 1)
        assert [v.id for v in items] == [1]
