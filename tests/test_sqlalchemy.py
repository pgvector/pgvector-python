import asyncpg
import numpy as np
import os
from pgvector import HalfVector, SparseVector, Vector
from pgvector.sqlalchemy import VECTOR, HALFVEC, BIT, SPARSEVEC, avg, sum
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

psycopg2_engine = create_engine('postgresql+psycopg2://localhost/pgvector_python_test')
psycopg2_type_engine = create_engine('postgresql+psycopg2://localhost/pgvector_python_test')


@event.listens_for(psycopg2_type_engine, "connect")
def psycopg2_connect(dbapi_connection, connection_record):
    from pgvector.psycopg2 import register_vector
    register_vector(dbapi_connection)


pg8000_engine = create_engine(f'postgresql+pg8000://{os.environ["USER"]}@localhost/pgvector_python_test')

if sqlalchemy_version > 1:
    psycopg_engine = create_engine('postgresql+psycopg://localhost/pgvector_python_test')
    psycopg_type_engine = create_engine('postgresql+psycopg://localhost/pgvector_python_test')

    @event.listens_for(psycopg_type_engine, "connect")
    def psycopg_connect(dbapi_connection, connection_record):
        from pgvector.psycopg import register_vector
        register_vector(dbapi_connection)

    psycopg_async_engine = create_async_engine('postgresql+psycopg://localhost/pgvector_python_test')
    psycopg_async_type_engine = create_async_engine('postgresql+psycopg://localhost/pgvector_python_test')

    @event.listens_for(psycopg_async_type_engine.sync_engine, "connect")
    def psycopg_async_connect(dbapi_connection, connection_record):
        from pgvector.psycopg import register_vector_async
        dbapi_connection.run_async(register_vector_async)

    asyncpg_engine = create_async_engine('postgresql+asyncpg://localhost/pgvector_python_test')
    asyncpg_type_engine = create_async_engine('postgresql+asyncpg://localhost/pgvector_python_test')

    @event.listens_for(asyncpg_type_engine.sync_engine, "connect")
    def asyncpg_connect(dbapi_connection, connection_record):
        from pgvector.asyncpg import register_vector
        dbapi_connection.run_async(register_vector)

engines = [psycopg2_engine, psycopg2_type_engine, pg8000_engine]
array_engines = [psycopg2_type_engine]
async_engines = []
async_array_engines = []

if sqlalchemy_version > 1:
    engines += [psycopg_engine, psycopg_type_engine]
    array_engines += [psycopg_type_engine]
    # TODO support asyncpg_type_engine
    async_engines += [psycopg_async_engine, psycopg_async_type_engine, asyncpg_engine]
    async_array_engines += [psycopg_async_type_engine, asyncpg_engine]

setup_engine = engines[0]
with Session(setup_engine) as session:
    session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    session.commit()

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


Base.metadata.drop_all(setup_engine)
Base.metadata.create_all(setup_engine)

index = Index(
    'sqlalchemy_orm_index',
    Item.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'vector_l2_ops'}
)
index.create(setup_engine)

half_precision_index = Index(
    'sqlalchemy_orm_half_precision_index',
    func.cast(Item.embedding, HALFVEC(3)).label('embedding'),
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'halfvec_l2_ops'}
)
half_precision_index.create(setup_engine)

binary_quantize_index = Index(
    'sqlalchemy_orm_binary_quantize_index',
    func.cast(func.binary_quantize(Item.embedding), BIT(3)).label('embedding'),
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'bit_hamming_ops'}
)
binary_quantize_index.create(setup_engine)


def create_items():
    with Session(setup_engine) as session:
        session.add(Item(id=1, embedding=[1, 1, 1], half_embedding=[1, 1, 1], binary_embedding='000', sparse_embedding=SparseVector([1, 1, 1])))
        session.add(Item(id=2, embedding=[2, 2, 2], half_embedding=[2, 2, 2], binary_embedding='101', sparse_embedding=SparseVector([2, 2, 2])))
        session.add(Item(id=3, embedding=[1, 1, 2], half_embedding=[1, 1, 2], binary_embedding='111', sparse_embedding=SparseVector([1, 1, 2])))
        session.commit()


def delete_items():
    with Session(setup_engine) as session:
        session.query(Item).delete()
        session.commit()


@pytest.mark.parametrize('engine', engines)
class TestSqlalchemy:
    def setup_method(self):
        delete_items()

    def test_core(self, engine):
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

    def test_orm(self, engine):
        item = Item(embedding=np.array([1.5, 2, 3]))
        item2 = Item(embedding=[4, 5, 6])
        item3 = Item()

        with Session(engine) as session:
            session.add(item)
            session.add(item2)
            session.add(item3)
            session.commit()

        stmt = select(Item)
        with Session(engine) as session:
            items = [v[0] for v in session.execute(stmt).all()]
            # TODO improve
            assert items[0].id % 3 == 1
            assert items[1].id % 3 == 2
            assert items[2].id % 3 == 0
            assert np.array_equal(items[0].embedding, np.array([1.5, 2, 3]))
            assert items[0].embedding.dtype == np.float32
            assert np.array_equal(items[1].embedding, np.array([4, 5, 6]))
            assert items[1].embedding.dtype == np.float32
            assert items[2].embedding is None

    def test_vector(self, engine):
        with Session(engine) as session:
            session.add(Item(id=1, embedding=[1, 2, 3]))
            session.commit()
            item = session.get(Item, 1)
            assert np.array_equal(item.embedding, [1, 2, 3])

    def test_vector_l2_distance(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.embedding.l2_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_vector_l2_distance_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.embedding.l2_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_vector_max_inner_product(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.embedding.max_inner_product([1, 1, 1])).all()
            assert [v.id for v in items] == [2, 3, 1]

    def test_vector_max_inner_product_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.embedding.max_inner_product([1, 1, 1])))
            assert [v.id for v in items] == [2, 3, 1]

    def test_vector_cosine_distance(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.embedding.cosine_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 2, 3]

    def test_vector_cosine_distance_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.embedding.cosine_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 2, 3]

    def test_vector_l1_distance(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.embedding.l1_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_vector_l1_distance_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.embedding.l1_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_halfvec(self, engine):
        with Session(engine) as session:
            session.add(Item(id=1, half_embedding=[1, 2, 3]))
            session.commit()
            item = session.get(Item, 1)
            assert item.half_embedding == HalfVector([1, 2, 3])

    def test_halfvec_l2_distance(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.half_embedding.l2_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_halfvec_l2_distance_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.half_embedding.l2_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_halfvec_max_inner_product(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.half_embedding.max_inner_product([1, 1, 1])).all()
            assert [v.id for v in items] == [2, 3, 1]

    def test_halfvec_max_inner_product_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.half_embedding.max_inner_product([1, 1, 1])))
            assert [v.id for v in items] == [2, 3, 1]

    def test_halfvec_cosine_distance(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.half_embedding.cosine_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 2, 3]

    def test_halfvec_cosine_distance_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.half_embedding.cosine_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 2, 3]

    def test_halfvec_l1_distance(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.half_embedding.l1_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_halfvec_l1_distance_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.half_embedding.l1_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_bit(self, engine):
        with Session(engine) as session:
            session.add(Item(id=1, binary_embedding='101'))
            session.commit()
            item = session.get(Item, 1)
            assert item.binary_embedding == '101'

    def test_bit_hamming_distance(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.binary_embedding.hamming_distance('101')).all()
            assert [v.id for v in items] == [2, 3, 1]

    def test_bit_hamming_distance_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.binary_embedding.hamming_distance('101')))
            assert [v.id for v in items] == [2, 3, 1]

    def test_bit_jaccard_distance(self, engine):
        if engine == pg8000_engine:
            return

        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.binary_embedding.jaccard_distance('101')).all()
            assert [v.id for v in items] == [2, 3, 1]

    def test_bit_jaccard_distance_orm(self, engine):
        if engine == pg8000_engine:
            return

        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.binary_embedding.jaccard_distance('101')))
            assert [v.id for v in items] == [2, 3, 1]

    def test_sparsevec(self, engine):
        with Session(engine) as session:
            session.add(Item(id=1, sparse_embedding=[1, 2, 3]))
            session.commit()
            item = session.get(Item, 1)
            assert item.sparse_embedding == SparseVector([1, 2, 3])

    def test_sparsevec_l2_distance(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.sparse_embedding.l2_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_sparsevec_l2_distance_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.sparse_embedding.l2_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_sparsevec_max_inner_product(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.sparse_embedding.max_inner_product([1, 1, 1])).all()
            assert [v.id for v in items] == [2, 3, 1]

    def test_sparsevec_max_inner_product_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.sparse_embedding.max_inner_product([1, 1, 1])))
            assert [v.id for v in items] == [2, 3, 1]

    def test_sparsevec_cosine_distance(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.sparse_embedding.cosine_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 2, 3]

    def test_sparsevec_cosine_distance_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.sparse_embedding.cosine_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 2, 3]

    def test_sparsevec_l1_distance(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(Item.sparse_embedding.l1_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_sparsevec_l1_distance_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).order_by(Item.sparse_embedding.l1_distance([1, 1, 1])))
            assert [v.id for v in items] == [1, 3, 2]

    def test_filter(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).filter(Item.embedding.l2_distance([1, 1, 1]) < 1).all()
            assert [v.id for v in items] == [1]

    def test_filter_orm(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.scalars(select(Item).filter(Item.embedding.l2_distance([1, 1, 1]) < 1))
            assert [v.id for v in items] == [1]

    def test_select(self, engine):
        with Session(engine) as session:
            session.add(Item(embedding=[2, 3, 3]))
            items = session.query(Item.embedding.l2_distance([1, 1, 1])).first()
            assert items[0] == 3

    def test_select_orm(self, engine):
        with Session(engine) as session:
            session.add(Item(embedding=[2, 3, 3]))
            items = session.scalars(select(Item.embedding.l2_distance([1, 1, 1]))).all()
            assert items[0] == 3

    def test_avg(self, engine):
        with Session(engine) as session:
            res = session.query(avg(Item.embedding)).first()[0]
            assert res is None
            session.add(Item(embedding=[1, 2, 3]))
            session.add(Item(embedding=[4, 5, 6]))
            res = session.query(avg(Item.embedding)).first()[0]
            assert np.array_equal(res, np.array([2.5, 3.5, 4.5]))

    def test_avg_orm(self, engine):
        with Session(engine) as session:
            res = session.scalars(select(avg(Item.embedding))).first()
            assert res is None
            session.add(Item(embedding=[1, 2, 3]))
            session.add(Item(embedding=[4, 5, 6]))
            res = session.scalars(select(avg(Item.embedding))).first()
            assert np.array_equal(res, np.array([2.5, 3.5, 4.5]))

    def test_sum(self, engine):
        with Session(engine) as session:
            res = session.query(sum(Item.embedding)).first()[0]
            assert res is None
            session.add(Item(embedding=[1, 2, 3]))
            session.add(Item(embedding=[4, 5, 6]))
            res = session.query(sum(Item.embedding)).first()[0]
            assert np.array_equal(res, np.array([5, 7, 9]))

    def test_sum_orm(self, engine):
        with Session(engine) as session:
            res = session.scalars(select(sum(Item.embedding))).first()
            assert res is None
            session.add(Item(embedding=[1, 2, 3]))
            session.add(Item(embedding=[4, 5, 6]))
            res = session.scalars(select(sum(Item.embedding))).first()
            assert np.array_equal(res, np.array([5, 7, 9]))

    def test_bad_dimensions(self, engine):
        item = Item(embedding=[1, 2])
        with Session(engine) as session:
            session.add(item)
            with pytest.raises(StatementError, match='expected 3 dimensions, not 2'):
                session.commit()

    def test_bad_ndim(self, engine):
        item = Item(embedding=np.array([[1, 2, 3]]))
        with Session(engine) as session:
            session.add(item)
            with pytest.raises(StatementError, match='expected ndim to be 1'):
                session.commit()

    def test_bad_dtype(self, engine):
        item = Item(embedding=np.array(['one', 'two', 'three']))
        with Session(engine) as session:
            session.add(item)
            with pytest.raises(StatementError, match='could not convert string to float'):
                session.commit()

    def test_inspect(self, engine):
        columns = inspect(engine).get_columns('sqlalchemy_orm_item')
        assert isinstance(columns[1]['type'], VECTOR)

    def test_literal_binds(self, engine):
        sql = select(Item).order_by(Item.embedding.l2_distance([1, 2, 3])).compile(engine, compile_kwargs={'literal_binds': True})
        assert "embedding <-> '[1.0,2.0,3.0]'" in str(sql)

    def test_insert(self, engine):
        with Session(engine) as session:
            session.execute(insert(Item).values(embedding=np.array([1, 2, 3])))

    def test_insert_bulk(self, engine):
        with Session(engine) as session:
            session.execute(insert(Item), [{'embedding': np.array([1, 2, 3])}])

    # register_vector in psycopg2 tests change this behavior
    # def test_insert_text(self):
    #     with Session(engine) as session:
    #         session.execute(text('INSERT INTO sqlalchemy_orm_item (embedding) VALUES (:embedding)'), {'embedding': np.array([1, 2, 3])})

    def test_automap(self, engine):
        metadata = MetaData()
        metadata.reflect(engine, only=['sqlalchemy_orm_item'])
        AutoBase = automap_base(metadata=metadata)
        AutoBase.prepare()
        AutoItem = AutoBase.classes.sqlalchemy_orm_item
        with Session(engine) as session:
            session.execute(insert(AutoItem), [{'embedding': np.array([1, 2, 3])}])
            item = session.query(AutoItem).first()
            assert np.array_equal(item.embedding, [1, 2, 3])

    def test_half_precision(self, engine):
        create_items()
        with Session(engine) as session:
            items = session.query(Item).order_by(func.cast(Item.embedding, HALFVEC(3)).l2_distance([1, 1, 1])).all()
            assert [v.id for v in items] == [1, 3, 2]

    def test_binary_quantize(self, engine):
        with Session(engine) as session:
            session.add(Item(id=1, embedding=[-1, -2, -3]))
            session.add(Item(id=2, embedding=[1, -2, 3]))
            session.add(Item(id=3, embedding=[1, 2, 3]))
            session.commit()

            distance = func.cast(func.binary_quantize(Item.embedding), BIT(3)).hamming_distance(func.binary_quantize(func.cast([3, -1, 2], VECTOR(3))))
            items = session.query(Item).order_by(distance).all()
            assert [v.id for v in items] == [2, 3, 1]


@pytest.mark.parametrize('engine', array_engines)
class TestSqlalchemyArray:
    def setup_method(self):
        delete_items()

    def test_vector_array(self, engine):
        with Session(engine) as session:
            session.add(Item(id=1, embeddings=[np.array([1, 2, 3]), np.array([4, 5, 6])]))
            session.commit()

            # this fails if the driver does not cast arrays
            item = session.get(Item, 1)
            assert np.array_equal(item.embeddings[0], [1, 2, 3])
            assert np.array_equal(item.embeddings[1], [4, 5, 6])

    def test_halfvec_array(self, engine):
        with Session(engine) as session:
            session.add(Item(id=1, half_embeddings=[np.array([1, 2, 3]), np.array([4, 5, 6])]))
            session.commit()

            # this fails if the driver does not cast arrays
            item = session.get(Item, 1)
            assert item.half_embeddings == [HalfVector([1, 2, 3]), HalfVector([4, 5, 6])]


@pytest.mark.parametrize('engine', async_engines)
class TestSqlalchemyAsync:
    def setup_method(self):
        delete_items()

    @pytest.mark.asyncio
    async def test_vector(self, engine):
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as session:
            async with session.begin():
                embedding = np.array([1, 2, 3])
                session.add(Item(id=1, embedding=embedding))
                item = await session.get(Item, 1)
                assert np.array_equal(item.embedding, embedding)

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_halfvec(self, engine):
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as session:
            async with session.begin():
                embedding = [1, 2, 3]
                session.add(Item(id=1, half_embedding=embedding))
                item = await session.get(Item, 1)
                assert item.half_embedding == HalfVector(embedding)

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_bit(self, engine):
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as session:
            async with session.begin():
                embedding = '101'
                session.add(Item(id=1, binary_embedding=embedding))
                item = await session.get(Item, 1)
                assert item.binary_embedding == embedding

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_sparsevec(self, engine):
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as session:
            async with session.begin():
                embedding = [1, 2, 3]
                session.add(Item(id=1, sparse_embedding=embedding))
                item = await session.get(Item, 1)
                assert item.sparse_embedding == SparseVector(embedding)

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_avg(self, engine):
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as session:
            async with session.begin():
                session.add(Item(embedding=[1, 2, 3]))
                session.add(Item(embedding=[4, 5, 6]))
                res = await session.scalars(select(avg(Item.embedding)))
                assert np.array_equal(res.first(), [2.5, 3.5, 4.5])

        await engine.dispose()


@pytest.mark.parametrize('engine', async_array_engines)
class TestSqlalchemyAsyncArray:
    def setup_method(self):
        delete_items()

    @pytest.mark.asyncio
    async def test_vector_array(self, engine):
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with async_session() as session:
            async with session.begin():
                session.add(Item(id=1, embeddings=[Vector([1, 2, 3]), Vector([4, 5, 6])]))
                item = await session.get(Item, 1)
                assert np.array_equal(item.embeddings[0], [1, 2, 3])
                assert np.array_equal(item.embeddings[1], [4, 5, 6])

                session.add(Item(id=2, embeddings=[np.array([1, 2, 3]), np.array([4, 5, 6])]))
                item = await session.get(Item, 2)
                assert np.array_equal(item.embeddings[0], [1, 2, 3])
                assert np.array_equal(item.embeddings[1], [4, 5, 6])

        await engine.dispose()
