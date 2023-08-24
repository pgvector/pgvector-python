from typing import Any

import numpy as np
import pytest
from peewee import (
    AutoField,
    DatabaseProxy,
    Model,
    PostgresqlDatabase,
)

from pgvector.peewee import VectorField


db_proxy = DatabaseProxy()

class BaseModel(Model):
    class Meta:
        database = db_proxy


class Item(BaseModel):
    id = AutoField(primary_key=True)
    embedding = VectorField()



@pytest.fixture(name='pg_database', autouse=True)
def fixture_pg_database():
    db = PostgresqlDatabase('pgvector_python_test', host='localhost')
    db_proxy.initialize(db)

    models = BaseModel.__subclasses__()
    db.bind(models, bind_refs=False, bind_backrefs=False)

    db.connect()
    db.create_tables(models)

    db.execute_sql('CREATE EXTENSION IF NOT EXISTS vector')

    for model in models:
        model.truncate_table()

    yield db

    for model in models:
        model.truncate_table()

    db.close()



def test_select(pg_database: PostgresqlDatabase):
    assert pg_database.execute_sql('SELECT 1').fetchone() == (1,)


@pytest.mark.parametrize('vector', [[1, 2, 3], np.array([1, 2, 3])], ids=['list', 'np'])
def test_create_vector(vector: Any):
    item = Item.create(embedding=vector)
    assert Item.get_by_id(item.id) == item


def test_l2_distance_sort():
    item1 = Item.create(embedding=[1, 2, 3])
    item2 = Item.create(embedding=[4, 5, 6])
    item3 = Item.create(embedding=[0, 0, 0])

    query = (
        Item.select()
        .where(Item.id != item1.id)
        .order_by(Item.embedding.l2_distance(item1.embedding).desc())
    )

    items = list(query)
    assert items == [item2, item3]


@pytest.mark.parametrize(
    argnames=['method', 'expected'],
    argvalues=[
        (Item.embedding.l2_distance, 5.196152422706632),
        (Item.embedding.cosine_distance, 0.025368153802923787),
        (Item.embedding.max_inner_product, -32.0),
    ],
    ids=['l2_distance', 'cosine_distance', 'max_inner_prod'],
)
def test_distance_measures(method: Any, expected: float):
    item = Item.create(embedding=[1, 2, 3])
    vector = [4, 5, 6]

    measure = Item.select(method(vector).alias('dist')).where(Item.id == item.id).get()
    assert measure.dist == expected
