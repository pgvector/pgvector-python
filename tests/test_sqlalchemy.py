import numpy as np
from pgvector.sqlalchemy import VECTOR, HALFVEC, BIT, SPARSEVEC, SparseVector, avg, sum
import pytest
from sqlalchemy import create_engine, event, insert, inspect, select, text, MetaData, Table, Column, Index, Integer, ARRAY
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.sql import func

try:
    from sqlalchemy.orm import mapped_column
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    sqlalchemy_version = 2
except ImportError:
    mapped_column = Column
    sqlalchemy_version = 1

engine = create_engine('postgresql+psycopg2://localhost/pgvector_python_test')
with Session(engine) as session:
    session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    session.commit()

array_engine = create_engine('postgresql+psycopg2://localhost/pgvector_python_test')


@event.listens_for(array_engine, "connect")
def connect(dbapi_connection, connection_record):
    from pgvector.psycopg2 import register_vector
    register_vector(dbapi_connection, globally=False, arrays=True)


Base = declarative_base()


class Item(Base):
    __tablename__ = 'sqlalchemy_orm_item'

    id = mapped_column(Integer, primary_key=True)
    embedding = mapped_column(VECTOR(3))
    half_embedding = mapped_column(HALFVEC(3))
    binary_embedding = mapped_column(BIT(3))
    sparse_embedding = mapped_column(SPARSEVEC(3))
    embeddings = mapped_column(ARRAY(VECTOR(3)))
    half_embeddings = mapped_column(ARRAY(HALFVEC(3)))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

index = Index(
    'sqlalchemy_orm_index',
    Item.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'vector_l2_ops'}
)
index.create(engine)

half_precision_index = Index(
    'sqlalchemy_orm_half_precision_index',
    func.cast(Item.embedding, HALFVEC(3)).label('embedding'),
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'halfvec_l2_ops'}
)
half_precision_index.create(engine)

binary_quantize_index = Index(
    'sqlalchemy_orm_binary_quantize_index',
    func.cast(func.binary_quantize(Item.embedding), BIT(3)).label('embedding'),
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'bit_hamming_ops'}
)
binary_quantize_index.create(engine)


def create_items():
    session = Session(engine)
    session.add(Item(id=1, embedding=[1, 1, 1], half_embedding=[1, 1, 1], binary_embedding='000', sparse_embedding=SparseVector([1, 1, 1])))
    session.add(Item(id=2, embedding=[2, 2, 2], half_embedding=[2, 2, 2], binary_embedding='101', sparse_embedding=SparseVector([2, 2, 2])))
    session.add(Item(id=3, embedding=[1, 1, 2], half_embedding=[1, 1, 2], binary_embedding='111', sparse_embedding=SparseVector([1, 1, 2])))
    session.commit()


class TestSqlalchemy:
    def setup_method(self, test_method):
        with Session(engine) as session:
            session.query(Item).delete()
            session.commit()

    def test_core(self):
        metadata = MetaData()

        item_table = Table(
            'sqlalchemy_core_item',
            metadata,
            Column('id', Integer, primary_key=True),
            Column('embedding', VECTOR(3)),
            Column('half_embedding', HALFVEC(3)),
            Column('binary_embedding', BIT(3)),
            Column('sparse_embedding', SPARSEVEC(3)),
            Column('embeddings', ARRAY(VECTOR(3)))
        )

        metadata.drop_all(engine)
        metadata.create_all(engine)

        ivfflat_index = Index(
            'sqlalchemy_core_ivfflat_index',
            item_table.c.embedding,
            postgresql_using='ivfflat',
            postgresql_with={'lists': 1},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        )
        ivfflat_index.create(engine)

        hnsw_index = Index(
            'sqlalchemy_core_hnsw_index',
            item_table.c.embedding,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        )
        hnsw_index.create(engine)

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

    def test_vector(self):
        session = Session(engine)
        session.add(Item(id=1, embedding=[1, 2, 3]))
        session.commit()
        item = session.get(Item, 1)
        assert item.embedding.tolist() == [1, 2, 3]

    def test_vector_l2_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.embedding.l2_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_vector_l2_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.embedding.l2_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_vector_max_inner_product(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.embedding.max_inner_product([1, 1, 1])).all()
            assert [v.id for v in items] == [2, 3, 1]

    def test_vector_max_inner_product_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.embedding.max_inner_product([1, 1, 1])))
            assert [v.id for v in items] == [2, 3, 1]

    def test_vector_cosine_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.embedding.cosine_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 2, 3]

    def test_vector_cosine_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.embedding.cosine_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 2, 3]

    def test_vector_l1_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.embedding.l1_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_vector_l1_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.embedding.l1_distance([1, 1, 1])))
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
            items = session.query(Item).order_by(Item.half_embedding.l2_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_halfvec_l2_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.half_embedding.l2_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_halfvec_max_inner_product(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.half_embedding.max_inner_product([1, 1, 1])).all()
            assert [v.id for v in items] == [2, 3, 1]

    def test_halfvec_max_inner_product_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.half_embedding.max_inner_product([1, 1, 1])))
            assert [v.id for v in items] == [2, 3, 1]

    def test_halfvec_cosine_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.half_embedding.cosine_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 2, 3]

    def test_halfvec_cosine_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.half_embedding.cosine_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 2, 3]

    def test_halfvec_l1_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.half_embedding.l1_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_halfvec_l1_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.half_embedding.l1_distance([1, 1, 1])))
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
            items = session.query(Item).order_by(Item.binary_embedding.hamming_distance('101')).all()
            assert [v.id for v in items] == [2, 3, 1]

    def test_bit_hamming_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.binary_embedding.hamming_distance('101')))
            assert [v.id for v in items] == [2, 3, 1]

    def test_bit_jaccard_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.binary_embedding.jaccard_distance('101')).all()
            assert [v.id for v in items] == [2, 3, 1]

    def test_bit_jaccard_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.binary_embedding.jaccard_distance('101')))
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
            items = session.query(Item).order_by(Item.sparse_embedding.l2_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_sparsevec_l2_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.sparse_embedding.l2_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_sparsevec_max_inner_product(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.sparse_embedding.max_inner_product([1, 1, 1])).all()
            assert [v.id for v in items] == [2, 3, 1]

    def test_sparsevec_max_inner_product_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.sparse_embedding.max_inner_product([1, 1, 1])))
            assert [v.id for v in items] == [2, 3, 1]

    def test_sparsevec_cosine_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.sparse_embedding.cosine_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 2, 3]

    def test_sparsevec_cosine_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.sparse_embedding.cosine_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 2, 3]

    def test_sparsevec_l1_distance(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.sparse_embedding.l1_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_sparsevec_l1_distance_orm(self):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.sparse_embedding.l1_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

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
            items = session.query(Item.embedding.l2_distance([1, 1, 1])).first()
            assert items[0] == 3

    def test_select_orm(self):
        with Session(engine) as session:
            session.add(Item(embedding=[2, 3, 3]))
            items = session.scalars(select(Item.embedding.l2_distance([1, 1, 1]))).all()
            assert items[0] == 3

    def test_avg(self):
        with Session(engine) as session:
            res = session.query(avg(Item.embedding)).first()[0]
            assert res is None
            session.add(Item(embedding=[1, 2, 3]))
            session.add(Item(embedding=[4, 5, 6]))
            res = session.query(avg(Item.embedding)).first()[0]
            assert np.array_equal(res, np.array([2.5, 3.5, 4.5]))

    def test_avg_orm(self):
        with Session(engine) as session:
            res = session.scalars(select(avg(Item.embedding))).first()
            assert res is None
            session.add(Item(embedding=[1, 2, 3]))
            session.add(Item(embedding=[4, 5, 6]))
            res = session.scalars(select(avg(Item.embedding))).first()
            assert np.array_equal(res, np.array([2.5, 3.5, 4.5]))

    def test_sum(self):
        with Session(engine) as session:
            res = session.query(sum(Item.embedding)).first()[0]
            assert res is None
            session.add(Item(embedding=[1, 2, 3]))
            session.add(Item(embedding=[4, 5, 6]))
            res = session.query(sum(Item.embedding)).first()[0]
            assert np.array_equal(res, np.array([5, 7, 9]))

    def test_sum_orm(self):
        with Session(engine) as session:
            res = session.scalars(select(sum(Item.embedding))).first()
            assert res is None
            session.add(Item(embedding=[1, 2, 3]))
            session.add(Item(embedding=[4, 5, 6]))
            res = session.scalars(select(sum(Item.embedding))).first()
            assert np.array_equal(res, np.array([5, 7, 9]))

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
        with pytest.raises(StatementError, match='could not convert string to float'):
            session.commit()

    def test_inspect(self):
        columns = inspect(engine).get_columns('sqlalchemy_orm_item')
        assert isinstance(columns[1]['type'], VECTOR)

    def test_literal_binds(self):
        sql = select(Item).order_by(Item.embedding.l2_distance([1, 2, 3])).compile(engine, compile_kwargs={'literal_binds': True})
        assert "embedding <-> '[1.0,2.0,3.0]'" in str(sql)

    def test_insert(self):
        session.execute(insert(Item).values(embedding=np.array([1, 2, 3])))

    def test_insert_bulk(self):
        session.execute(insert(Item), [{'embedding': np.array([1, 2, 3])}])

    # register_vector in psycopg2 tests change this behavior
    # def test_insert_text(self):
    #     session.execute(text('INSERT INTO sqlalchemy_orm_item (embedding) VALUES (:embedding)'), {'embedding': np.array([1, 2, 3])})

    def test_automap(self):
        metadata = MetaData()
        metadata.reflect(engine, only=['sqlalchemy_orm_item'])
        AutoBase = automap_base(metadata=metadata)
        AutoBase.prepare()
        AutoItem = AutoBase.classes.sqlalchemy_orm_item
        session.execute(insert(AutoItem), [{'embedding': np.array([1, 2, 3])}])
        item = session.query(AutoItem).first()
        assert item.embedding.tolist() == [1, 2, 3]

    def test_vector_array(self):
        session = Session(array_engine)
        session.add(Item(id=1, embeddings=[np.array([1, 2, 3]), np.array([4, 5, 6])]))
        session.commit()

        # this fails if the driver does not cast arrays
        item = session.get(Item, 1)
        assert item.embeddings[0].tolist() == [1, 2, 3]
        assert item.embeddings[1].tolist() == [4, 5, 6]

    def test_halfvec_array(self):
        session = Session(array_engine)
        session.add(Item(id=1, half_embeddings=[np.array([1, 2, 3]), np.array([4, 5, 6])]))
        session.commit()

        # this fails if the driver does not cast arrays
        item = session.get(Item, 1)
        assert item.half_embeddings[0].to_list() == [1, 2, 3]
        assert item.half_embeddings[1].to_list() == [4, 5, 6]

    def test_half_precision(self):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(func.cast(Item.embedding, HALFVEC(3)).l2_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_binary_quantize(self):
        session = Session(engine)
        session.add(Item(id=1, embedding=[-1, -2, -3]))
        session.add(Item(id=2, embedding=[1, -2, 3]))
        session.add(Item(id=3, embedding=[1, 2, 3]))
        session.commit()

        with Session(engine) as session:
            distance = func.cast(func.binary_quantize(Item.embedding), BIT(3)).hamming_distance(func.binary_quantize(func.cast([3, -1, 2], VECTOR(3))))
            items = session.query(Item).order_by(distance).all()
            assert [v.id for v in items] == [2, 3, 1]

    @pytest.mark.asyncio
    @pytest.mark.skipif(sqlalchemy_version == 1, reason='Requires SQLAlchemy 2+')
    async def test_async(self):
        engine = create_async_engine('postgresql+psycopg://localhost/pgvector_python_test')
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as session:
            async with session.begin():
                session.add(Item(embedding=[1, 2, 3]))
                session.add(Item(embedding=[4, 5, 6]))
                avg = await session.scalars(select(func.avg(Item.embedding)))
                assert avg.first() == '[2.5,3.5,4.5]'

        await engine.dispose()

    @pytest.mark.asyncio
    @pytest.mark.skipif(sqlalchemy_version == 1, reason='Requires SQLAlchemy 2+')
    async def test_async_vector_array(self):
        engine = create_async_engine('postgresql+psycopg://localhost/pgvector_python_test')
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        @event.listens_for(engine.sync_engine, "connect")
        def connect(dbapi_connection, connection_record):
            from pgvector.psycopg import register_vector_async
            dbapi_connection.run_async(register_vector_async)

        async with async_session() as session:
            async with session.begin():
                session.add(Item(id=1, embeddings=[np.array([1, 2, 3]), np.array([4, 5, 6])]))

                # this fails if the driver does not cast arrays
                item = await session.get(Item, 1)
                assert item.embeddings[0].tolist() == [1, 2, 3]
                assert item.embeddings[1].tolist() == [4, 5, 6]

        await engine.dispose()

    @pytest.mark.asyncio
    @pytest.mark.skipif(sqlalchemy_version == 1, reason='Requires SQLAlchemy 2+')
    async def test_asyncpg_bit(self):
        import asyncpg

        engine = create_async_engine('postgresql+asyncpg://localhost/pgvector_python_test')
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as session:
            async with session.begin():
                embedding = asyncpg.BitString('101')
                session.add(Item(id=1, binary_embedding=embedding))
                item = await session.get(Item, 1)
                assert item.binary_embedding == embedding

        await engine.dispose()
