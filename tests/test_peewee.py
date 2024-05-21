from math import sqrt
import numpy as np
from peewee import Model, PostgresqlDatabase, fn
from pgvector.peewee import VectorField, HalfVectorField, FixedBitField, SparseVectorField, SparseVector

db = PostgresqlDatabase('pgvector_python_test')


class BaseModel(Model):
    class Meta:
        database = db


class Item(BaseModel):
    embedding = VectorField(dimensions=3, null=True)
    half_embedding = HalfVectorField(dimensions=3, null=True)
    binary_embedding = FixedBitField(max_length=3, null=True)
    sparse_embedding = SparseVectorField(dimensions=3, null=True)

    class Meta:
        table_name = 'peewee_item'


Item.add_index('embedding vector_l2_ops', using='hnsw')

db.connect()
db.execute_sql('CREATE EXTENSION IF NOT EXISTS vector')
db.drop_tables([Item])
db.create_tables([Item])


def create_items():
    Item.create(id=1, embedding=[1, 1, 1], half_embedding=[1, 1, 1], binary_embedding='000', sparse_embedding=SparseVector.from_dense([1, 1, 1]))
    Item.create(id=2, embedding=[2, 2, 2], half_embedding=[2, 2, 2], binary_embedding='101', sparse_embedding=SparseVector.from_dense([2, 2, 2]))
    Item.create(id=3, embedding=[1, 1, 2], half_embedding=[1, 1, 2], binary_embedding='111', sparse_embedding=SparseVector.from_dense([1, 1, 2]))


class TestPeewee:
    def setup_method(self, test_method):
        Item.truncate_table()

    def test_vector(self):
        Item.create(id=1, embedding=[1, 2, 3])
        item = Item.get_by_id(1)
        assert np.array_equal(item.embedding, np.array([1, 2, 3]))
        assert item.embedding.dtype == np.float32

    def test_vector_l2_distance(self):
        create_items()
        distance = Item.embedding.l2_distance([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]

    def test_vector_max_inner_product(self):
        create_items()
        distance = Item.embedding.max_inner_product([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [-6, -4, -3]

    def test_vector_cosine_distance(self):
        create_items()
        distance = Item.embedding.cosine_distance([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [1, 2, 3]
        assert [v.distance for v in items] == [0, 0, 0.05719095841793653]

    def test_vector_l1_distance(self):
        create_items()
        distance = Item.embedding.l1_distance([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, 3]

    def test_halfvec_l2_distance(self):
        create_items()
        distance = Item.half_embedding.l2_distance([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]

    def test_halfvec_max_inner_product(self):
        create_items()
        distance = Item.half_embedding.max_inner_product([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [-6, -4, -3]

    def test_halfvec_cosine_distance(self):
        create_items()
        distance = Item.half_embedding.cosine_distance([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [1, 2, 3]
        assert [v.distance for v in items] == [0, 0, 0.05719095841793653]

    def test_halfvec_l1_distance(self):
        create_items()
        distance = Item.half_embedding.l1_distance([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, 3]

    def test_sparsevec_l2_distance(self):
        create_items()
        distance = Item.sparse_embedding.l2_distance(SparseVector.from_dense([1, 1, 1]))
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]

    def test_sparsevec_max_inner_product(self):
        create_items()
        distance = Item.sparse_embedding.max_inner_product([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [-6, -4, -3]

    def test_sparsevec_cosine_distance(self):
        create_items()
        distance = Item.sparse_embedding.cosine_distance([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [1, 2, 3]
        assert [v.distance for v in items] == [0, 0, 0.05719095841793653]

    def test_sparsevec_l1_distance(self):
        create_items()
        distance = Item.sparse_embedding.l1_distance([1, 1, 1])
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, 3]

    def test_bit_hamming_distance(self):
        create_items()
        distance = Item.binary_embedding.hamming_distance('101')
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [0, 1, 2]

    def test_bit_jaccard_distance(self):
        create_items()
        distance = Item.binary_embedding.jaccard_distance('101')
        items = Item.select(Item.id, distance.alias('distance')).order_by(distance).limit(5)
        assert [v.id for v in items] == [2, 3, 1]
        # assert [v.distance for v in items] == [0, 1/3, 1]

    def test_where(self):
        create_items()
        items = Item.select().where(Item.embedding.l2_distance([1, 1, 1]) < 1)
        assert [v.id for v in items] == [1]

    def test_vector_avg(self):
        avg = Item.select(fn.avg(Item.embedding)).scalar()
        assert avg is None
        Item.create(embedding=[1, 2, 3])
        Item.create(embedding=[4, 5, 6])
        avg = Item.select(fn.avg(Item.embedding)).scalar()
        assert np.array_equal(avg, np.array([2.5, 3.5, 4.5]))

    def test_vector_sum(self):
        sum = Item.select(fn.sum(Item.embedding)).scalar()
        assert sum is None
        Item.create(embedding=[1, 2, 3])
        Item.create(embedding=[4, 5, 6])
        sum = Item.select(fn.sum(Item.embedding)).scalar()
        assert np.array_equal(sum, np.array([5, 7, 9]))

    def test_halfvec_avg(self):
        avg = Item.select(fn.avg(Item.half_embedding)).scalar()
        assert avg is None
        Item.create(half_embedding=[1, 2, 3])
        Item.create(half_embedding=[4, 5, 6])
        avg = Item.select(fn.avg(Item.half_embedding)).scalar()
        assert avg.to_list() == [2.5, 3.5, 4.5]

    def test_halfvec_sum(self):
        sum = Item.select(fn.sum(Item.half_embedding)).scalar()
        assert sum is None
        Item.create(half_embedding=[1, 2, 3])
        Item.create(half_embedding=[4, 5, 6])
        sum = Item.select(fn.sum(Item.half_embedding)).scalar()
        assert sum.to_list() == [5, 7, 9]

    def test_get_or_create(self):
        Item.get_or_create(id=1, defaults={'embedding': [1, 2, 3]})
        Item.get_or_create(embedding=np.array([4, 5, 6]))
        Item.get_or_create(embedding=Item.embedding.to_value([7, 8, 9]))
