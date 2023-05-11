import numpy as np
from pgvector.sqlalchemy import Vector
import pytest
from sqlalchemy import create_engine, select, text, MetaData, Table, Column, Index, Integer
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import declarative_base, mapped_column, Session

engine = create_engine('postgresql+psycopg2://localhost/pgvector_python_test')
with engine.connect() as con:
    con.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    con.commit()

Base = declarative_base()


class Item(Base):
    __tablename__ = 'orm_item'

    id = mapped_column(Integer, primary_key=True)
    embedding = mapped_column(Vector(3))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)


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


class TestSqlalchemy:
    def setup_method(self, test_method):
        with Session(engine) as session:
            session.query(Item).delete()
            session.commit()

    def test_core(self):
        metadata = MetaData()

        item_table = Table(
            'core_item',
            metadata,
            Column('id', Integer, primary_key=True),
            Column('embedding', Vector(3))
        )

        metadata.drop_all(engine)
        metadata.create_all(engine)

        index = Index(
            'my_core_index',
            item_table.c.embedding,
            postgresql_using='ivfflat',
            postgresql_with={'lists': 1},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        )
        index.create(engine)

    def test_orm(self):
        item = Item(embedding=np.array([1.5, 2, 3]))
        item2 = Item(embedding=[4, 5, 6])
        item3 = Item()

        session = Session(engine)
        session.add(item)
        session.add(item2)
        session.add(item3)
        session.commit()

        stmt = select(Item)
        with Session(engine) as session:
            items = [v[0] for v in session.execute(stmt).all()]
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
            items = session.query(Item).order_by(Item.embedding.l2_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_l2_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.embedding.l2_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_max_inner_product(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.embedding.max_inner_product([1, 1, 1])).all()
            assert [v.id for v in items] == [2, 3, 1]

    def test_max_inner_product_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.embedding.max_inner_product([1, 1, 1])))
            assert [v.id for v in items] == [2, 3, 1]

    def test_cosine_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.embedding.cosine_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 2, 3]

    def test_cosine_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.embedding.cosine_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 2, 3]

    def test_filter(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).filter(Item.embedding.l2_distance([1, 1, 1]) < 1).all()
            assert [v.id for v in items] == [1]

    def test_filter_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).filter(Item.embedding.l2_distance([1, 1, 1]) < 1))
            assert [v.id for v in items] == [1]

    def test_select(self):
        with Session(engine) as session:
            session.add(Item(embedding=[2, 3, 3]))
            item = session.query(Item.embedding.l2_distance([1, 1, 1])).first()
            assert item[0] == 3

    def test_bad_dimensions(self):
        item = Item(embedding=[1, 2])
        session = Session(engine)
        session.add(item)
        with pytest.raises(StatementError, match='expected 3 dimensions, not 2'):
            session.commit()

    def test_bad_ndim(self):
        item = Item(embedding=np.array([[1, 2, 3]]))
        session = Session(engine)
        session.add(item)
        with pytest.raises(StatementError, match='expected ndim to be 1'):
            session.commit()

    def test_bad_dtype(self):
        item = Item(embedding=np.array(['one', 'two', 'three']))
        session = Session(engine)
        session.add(item)
        with pytest.raises(StatementError, match='dtype must be numeric'):
            session.commit()
