import numpy as np
from pgvector.sqlalchemy import VECTOR, HALFVEC, BIT, SPARSEVEC, SparseVector
import pytest
from sqlalchemy import Column, Index
from sqlalchemy.exc import StatementError
from sqlalchemy.sql import func
from sqlmodel import Field, Session, SQLModel, create_engine, delete, select, text
from typing import Any, Optional

engine = create_engine('postgresql+psycopg2://localhost/pgvector_python_test')
with Session(engine) as session:
    session.exec(text('CREATE EXTENSION IF NOT EXISTS vector'))


class Item(SQLModel, table=True):
    __tablename__ = 'sqlmodel_item'

    id: Optional[int] = Field(default=None, primary_key=True)
    embedding: Optional[Any] = Field(default=None, sa_column=Column(VECTOR(3)))
    half_embedding: Optional[Any] = Field(default=None, sa_column=Column(HALFVEC(3)))
    binary_embedding: Optional[Any] = Field(default=None, sa_column=Column(BIT(3)))
    sparse_embedding: Optional[Any] = Field(default=None, sa_column=Column(SPARSEVEC(3)))


SQLModel.metadata.drop_all(engine)
SQLModel.metadata.create_all(engine)

index = Index(
    'sqlmodel_index',
    Item.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'vector_l2_ops'}
)
index.create(engine)


def create_items():
    session = Session(engine)
    session.add(Item(id=1, embedding=[1, 1, 1], half_embedding=[1, 1, 1], binary_embedding='000', sparse_embedding=SparseVector([1, 1, 1])))
    session.add(Item(id=2, embedding=[2, 2, 2], half_embedding=[2, 2, 2], binary_embedding='101', sparse_embedding=SparseVector([2, 2, 2])))
    session.add(Item(id=3, embedding=[1, 1, 2], half_embedding=[1, 1, 2], binary_embedding='111', sparse_embedding=SparseVector([1, 1, 2])))
    session.commit()


class TestSqlmodel:
    def setup_method(self, test_method):
        with Session(engine) as session:
            session.exec(delete(Item))
            session.commit()

    def test_orm(self):
        item = Item(embedding=[1.5, 2, 3])
        item2 = Item(embedding=[4, 5, 6])
        item3 = Item()

        session = Session(engine)
        session.add(item)
        session.add(item2)
        session.add(item3)
        session.commit()

        stmt = select(Item)
        with Session(engine) as session:
            items = session.exec(stmt).all()
            assert items[0].id == 1
            assert items[1].id == 2
            assert items[2].id == 3
            assert np.array_equal(items[0].embedding, np.array([1.5, 2, 3]))
            assert items[0].embedding.dtype == np.float32
            assert np.array_equal(items[1].embedding, np.array([4, 5, 6]))
            assert items[1].embedding.dtype == np.float32
            assert items[2].embedding is None

    def test_vector(self):
        session = Session(engine)
        session.add(Item(id=1, embedding=[1, 2, 3]))
        session.commit()
        item = session.get(Item, 1)
        assert item.embedding.tolist() == [1, 2, 3]

    def test_vector_l2_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.embedding.l2_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_vector_max_inner_product(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.embedding.max_inner_product([1, 1, 1])))
            assert [v.id for v in items] == [2, 3, 1]

    def test_vector_cosine_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.embedding.cosine_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 2, 3]

    def test_vector_l1_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.embedding.l1_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_halfvec(self):
        session = Session(engine)
        session.add(Item(id=1, half_embedding=[1, 2, 3]))
        session.commit()
        item = session.get(Item, 1)
        assert item.half_embedding.to_list() == [1, 2, 3]

    def test_halfvec_l2_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.half_embedding.l2_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_halfvec_max_inner_product(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.half_embedding.max_inner_product([1, 1, 1])))
            assert [v.id for v in items] == [2, 3, 1]

    def test_halfvec_cosine_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.half_embedding.cosine_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 2, 3]

    def test_halfvec_l1_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.half_embedding.l1_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_bit(self):
        session = Session(engine)
        session.add(Item(id=1, binary_embedding='101'))
        session.commit()
        item = session.get(Item, 1)
        assert item.binary_embedding == '101'

    def test_bit_hamming_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.binary_embedding.hamming_distance('101')))
            assert [v.id for v in items] == [2, 3, 1]

    def test_bit_jaccard_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.binary_embedding.jaccard_distance('101')))
            assert [v.id for v in items] == [2, 3, 1]

    def test_sparsevec(self):
        session = Session(engine)
        session.add(Item(id=1, sparse_embedding=[1, 2, 3]))
        session.commit()
        item = session.get(Item, 1)
        assert item.sparse_embedding.to_list() == [1, 2, 3]

    def test_sparsevec_l2_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.sparse_embedding.l2_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_sparsevec_max_inner_product(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.sparse_embedding.max_inner_product([1, 1, 1])))
            assert [v.id for v in items] == [2, 3, 1]

    def test_sparsevec_cosine_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.sparse_embedding.cosine_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 2, 3]

    def test_sparsevec_l1_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.sparse_embedding.l1_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_filter(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).filter(Item.embedding.l2_distance([1, 1, 1]) < 1))
            assert [v.id for v in items] == [1]

    def test_select(self):
        with Session(engine) as session:
            session.add(Item(embedding=[2, 3, 3]))
            items = session.exec(select(Item.embedding.l2_distance([1, 1, 1]))).all()
            assert items[0] == 3

    def test_vector_avg(self):
        with Session(engine) as session:
            avg = session.exec(select(func.avg(Item.embedding))).first()
            assert avg is None
            session.add(Item(embedding=[1, 2, 3]))
            session.add(Item(embedding=[4, 5, 6]))
            avg = session.exec(select(func.avg(Item.embedding))).first()
            assert np.array_equal(avg, np.array([2.5, 3.5, 4.5]))

    def test_vector_sum(self):
        with Session(engine) as session:
            sum = session.exec(select(func.sum(Item.embedding))).first()
            assert sum is None
            session.add(Item(embedding=[1, 2, 3]))
            session.add(Item(embedding=[4, 5, 6]))
            sum = session.exec(select(func.sum(Item.embedding))).first()
            assert np.array_equal(sum, np.array([5, 7, 9]))

    def test_halfvec_avg(self):
        with Session(engine) as session:
            avg = session.exec(select(func.avg(Item.half_embedding))).first()
            assert avg is None
            session.add(Item(half_embedding=[1, 2, 3]))
            session.add(Item(half_embedding=[4, 5, 6]))
            avg = session.exec(select(func.avg(Item.half_embedding))).first()
            assert avg.to_list() == [2.5, 3.5, 4.5]

    def test_halfvec_sum(self):
        with Session(engine) as session:
            sum = session.exec(select(func.sum(Item.half_embedding))).first()
            assert sum is None
            session.add(Item(half_embedding=[1, 2, 3]))
            session.add(Item(half_embedding=[4, 5, 6]))
            sum = session.exec(select(func.sum(Item.half_embedding))).first()
            assert sum.to_list() == [5, 7, 9]

    def test_bad_dimensions(self):
        item = Item(embedding=[1, 2])
        session = Session(engine)
        session.add(item)
        with pytest.raises(StatementError, match='expected 3 dimensions, not 2'):
            session.commit()
