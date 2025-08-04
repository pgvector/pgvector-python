from math import sqrt

import numpy as np

from tests.test_tortoise.models import Item
from pgvector.tortoise import (
    CosineDistanceAnnotation,
    MaxInnerProductAnnotation,
    L2DistanceAnnotation,
    L1DistanceAnnotation,
)
import pytest


from tortoise import Tortoise


async def create_items():
    await Item.create(id=1, embedding=[1, 1, 1])
    await Item.create(id=2, embedding=[2, 2, 2])
    await Item.create(id=3, embedding=[1, 1, 2])


class TestTortoise:
    @pytest.fixture(autouse=True)
    async def setup_db(self):
        await Tortoise.init(
            config={
                "connections": {
                    "default": {
                        "engine": "tortoise.backends.asyncpg",
                        "credentials": {
                            "database": "pgvector_python_test",
                            "host": ["localhost"],
                            "password": "password",
                            "port": "5432",
                            "user": "postgres",
                        },
                    },
                },
                "apps": {
                    "models": {
                        "models": ["tests.test_tortoise.models"],
                        "default_connection": "default",
                    }
                },
            }
        )

        conn = Tortoise.get_connection("default")
        await conn.execute_query("CREATE EXTENSION IF NOT EXISTS vector")
        await Tortoise.generate_schemas()
        await conn.execute_query("delete from tortoise_items")
        yield
        await Tortoise.close_connections()

    @pytest.mark.asyncio
    async def test_vector(self):
        await Item.create(id=1, embedding=[1, 2, 3])
        item = await Item.filter(id=1).first()
        assert np.array_equal(item.embedding, [1, 2, 3])
        assert item.embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_vector_l2_distance(self):
        await create_items()
        distance = L2DistanceAnnotation("embedding", [1, 1, 1])
        items = (
            await Item.all().annotate(distance=distance).order_by("distance").limit(5)
        )
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, sqrt(3)]

    @pytest.mark.asyncio
    async def test_vector_max_inner_product(self):
        await create_items()
        distance = MaxInnerProductAnnotation("embedding", [1, 1, 1])
        items = (
            await Item.all().annotate(distance=distance).order_by("distance").limit(5)
        )
        assert [v.id for v in items] == [2, 3, 1]
        assert [v.distance for v in items] == [-6, -4, -3]

    @pytest.mark.asyncio
    async def test_vector_cosine_distance(self):
        await create_items()
        distance = CosineDistanceAnnotation("embedding", [1, 1, 1])
        items = (
            await Item.all().annotate(distance=distance).order_by("distance").limit(5)
        )
        assert [v.id for v in items] == [1, 2, 3]
        assert [v.distance for v in items] == [0, 0, 0.05719095841793653]

    @pytest.mark.asyncio
    async def test_vector_l1_distance(self):
        await create_items()
        distance = L1DistanceAnnotation("embedding", [1, 1, 1])
        items = (
            await Item.all().annotate(distance=distance).order_by("distance").limit(5)
        )
        assert [v.id for v in items] == [1, 3, 2]
        assert [v.distance for v in items] == [0, 1, 3]
