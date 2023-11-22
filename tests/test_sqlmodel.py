import numpy as np
from pgvector.sqlalchemy import Vector
import pytest
from sqlalchemy import Column
from sqlalchemy.exc import StatementError
from sqlmodel import Field, Session, SQLModel, create_engine, delete, select, text
from typing import List, Optional

engine = create_engine('postgresql+psycopg2://localhost/pgvector_python_test')
with Session(engine) as session:
    session.exec(text('CREATE EXTENSION IF NOT EXISTS vector'))


class Item(SQLModel, table=True):
    __tablename__ = 'sqlmodel_item'

    id: Optional[int] = Field(default=None, primary_key=True)
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(3)))


SQLModel.metadata.drop_all(engine)
SQLModel.metadata.create_all(engine)


def create_items():
    vectors = [
        [1, 1, 1],
        [2, 2, 2],
        [1, 1, 2]
    ]
    session = Session(engine)
    for i, v in enumerate(vectors):
        session.add(Item(id=i + 1, embedding=v))
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

    def test_l2_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.embedding.l2_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_max_inner_product(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.embedding.max_inner_product([1, 1, 1])))
            assert [v.id for v in items] == [2, 3, 1]

    def test_cosine_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.exec(select(Item).order_by(Item.embedding.cosine_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 2, 3]

    def test_bad_dimensions(self):
        item = Item(embedding=[1, 2])
        session = Session(engine)
        session.add(item)
        with pytest.raises(StatementError, match='expected 3 dimensions, not 2'):
            session.commit()
