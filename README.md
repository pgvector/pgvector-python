# pgvector-python

[pgvector](https://github.com/pgvector/pgvector) support for Python

Supports [Django](https://github.com/django/django), [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy), [SQLModel](https://github.com/tiangolo/sqlmodel), [Psycopg 3](https://github.com/psycopg/psycopg), [Psycopg 2](https://github.com/psycopg/psycopg2), [asyncpg](https://github.com/MagicStack/asyncpg), [pg8000](https://github.com/tlocke/pg8000), and [Peewee](https://github.com/coleifer/peewee)

[![Build Status](https://github.com/pgvector/pgvector-python/actions/workflows/build.yml/badge.svg)](https://github.com/pgvector/pgvector-python/actions)

## Installation

Run:

```sh
pip install pgvector
```

And follow the instructions for your database library:

- [Django](#django)
- [SQLAlchemy](#sqlalchemy)
- [SQLModel](#sqlmodel)
- [Psycopg 3](#psycopg-3)
- [Psycopg 2](#psycopg-2)
- [asyncpg](#asyncpg)
- [pg8000](#pg8000)
- [Peewee](#peewee)

Or check out some examples:

- [Retrieval-augmented generation](https://github.com/pgvector/pgvector-python/blob/master/examples/rag/example.py) with Ollama
- [Embeddings](https://github.com/pgvector/pgvector-python/blob/master/examples/openai/example.py) with OpenAI
- [Binary embeddings](https://github.com/pgvector/pgvector-python/blob/master/examples/cohere/example.py) with Cohere
- [Sentence embeddings](https://github.com/pgvector/pgvector-python/blob/master/examples/sentence_transformers/example.py) with SentenceTransformers
- [Hybrid search](https://github.com/pgvector/pgvector-python/blob/master/examples/hybrid_search/rrf.py) with SentenceTransformers (Reciprocal Rank Fusion)
- [Hybrid search](https://github.com/pgvector/pgvector-python/blob/master/examples/hybrid_search/cross_encoder.py) with SentenceTransformers (cross-encoder)
- [Sparse search](https://github.com/pgvector/pgvector-python/blob/master/examples/sparse_search/example.py) with Transformers
- [Late interaction search](https://github.com/pgvector/pgvector-python/blob/master/examples/colbert/exact.py) with ColBERT
- [Visual document retrieval](https://github.com/pgvector/pgvector-python/blob/master/examples/colpali/exact.py) with ColPali
- [Image search](https://github.com/pgvector/pgvector-python/blob/master/examples/image_search/example.py) with PyTorch
- [Image search](https://github.com/pgvector/pgvector-python/blob/master/examples/imagehash/example.py) with perceptual hashing
- [Morgan fingerprints](https://github.com/pgvector/pgvector-python/blob/master/examples/rdkit/example.py) with RDKit
- [Topic modeling](https://github.com/pgvector/pgvector-python/blob/master/examples/gensim/example.py) with Gensim
- [Implicit feedback recommendations](https://github.com/pgvector/pgvector-python/blob/master/examples/implicit/example.py) with Implicit
- [Explicit feedback recommendations](https://github.com/pgvector/pgvector-python/blob/master/examples/surprise/example.py) with Surprise
- [Recommendations](https://github.com/pgvector/pgvector-python/blob/master/examples/lightfm/example.py) with LightFM
- [Horizontal scaling](https://github.com/pgvector/pgvector-python/blob/master/examples/citus/example.py) with Citus
- [Bulk loading](https://github.com/pgvector/pgvector-python/blob/master/examples/loading/example.py) with `COPY`

## Django

Create an empty migration file

```python
python manage.py makemigrations <your-app-name> --name enable_pgvector --empty
```

Add a migration in that file to enable the extension

```python
from pgvector.django import VectorExtension

class Migration(migrations.Migration):
    operations = [
        VectorExtension()
    ]
```

Migrate

```sh
python3 manage.py makemigrations
python3 manage.py migrate
```

Add a vector field in the models.py

```python
from pgvector.django import VectorField

class Item(models.Model):
    embedding = VectorField(dimensions=3)
```

Also supports `HalfVectorField`, `BitField`, and `SparseVectorField`

Migrate

```sh
python3 manage.py makemigrations
python3 manage.py migrate
```

Insert a vector

```python
item = Item(embedding=[1, 2, 3])
item.save()
```

Get the nearest neighbors to a vector

```python
from pgvector.django import L2Distance

Item.objects.order_by(L2Distance('embedding', [3, 1, 2]))[:5]
```

Also supports `MaxInnerProduct`, `CosineDistance`, `L1Distance`, `HammingDistance`, and `JaccardDistance`

Get the distance

```python
Item.objects.annotate(distance=L2Distance('embedding', [3, 1, 2]))
```

Get items within a certain distance

```python
Item.objects.alias(distance=L2Distance('embedding', [3, 1, 2])).filter(distance__lt=5)
```

Average vectors

```python
from django.db.models import Avg

Item.objects.aggregate(Avg('embedding'))
```

Also supports `Sum`

Add an approximate index

```python
from pgvector.django import HnswIndex, IvfflatIndex

class Item(models.Model):
    class Meta:
        indexes = [
            HnswIndex(
                name='my_index',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_l2_ops']
            ),
            # or
            IvfflatIndex(
                name='my_index',
                fields=['embedding'],
                lists=100,
                opclasses=['vector_l2_ops']
            )
        ]
```

Use `vector_ip_ops` for inner product and `vector_cosine_ops` for cosine distance

#### Half-Precision Indexing

Index vectors at half-precision

```python
from django.contrib.postgres.indexes import OpClass
from django.db.models.functions import Cast
from pgvector.django import HnswIndex, HalfVectorField

class Item(models.Model):
    class Meta:
        indexes = [
            HnswIndex(
                OpClass(Cast('embedding', HalfVectorField(dimensions=3)), name='halfvec_l2_ops'),
                name='my_index',
                m=16,
                ef_construction=64
            )
        ]
```

Note: Add `'django.contrib.postgres'` to `INSTALLED_APPS` to use `OpClass`

Get the nearest neighbors

```python
distance = L2Distance(Cast('embedding', HalfVectorField(dimensions=3)), [3, 1, 2])
Item.objects.order_by(distance)[:5]
```

## SQLAlchemy

Enable the extension

```python
session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
```

Add a vector column

```python
from pgvector.sqlalchemy import Vector

class Item(Base):
    embedding = mapped_column(Vector(3))
```

Also supports `HALFVEC`, `BIT`, and `SPARSEVEC`

Insert a vector

```python
item = Item(embedding=[1, 2, 3])
session.add(item)
session.commit()
```

Get the nearest neighbors to a vector

```python
session.scalars(select(Item).order_by(Item.embedding.l2_distance([3, 1, 2])).limit(5))
```

Also supports `max_inner_product`, `cosine_distance`, `l1_distance`, `hamming_distance`, and `jaccard_distance`

Get the distance

```python
session.scalars(select(Item.embedding.l2_distance([3, 1, 2])))
```

Get items within a certain distance

```python
session.scalars(select(Item).filter(Item.embedding.l2_distance([3, 1, 2]) < 5))
```

Average vectors

```python
from pgvector.sqlalchemy import avg

session.scalars(select(avg(Item.embedding))).first()
```

Also supports `sum`

Add an approximate index

```python
index = Index(
    'my_index',
    Item.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'vector_l2_ops'}
)
# or
index = Index(
    'my_index',
    Item.embedding,
    postgresql_using='ivfflat',
    postgresql_with={'lists': 100},
    postgresql_ops={'embedding': 'vector_l2_ops'}
)

index.create(engine)
```

Use `vector_ip_ops` for inner product and `vector_cosine_ops` for cosine distance

#### Half-Precision Indexing

Index vectors at half-precision

```python
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy.sql import func

index = Index(
    'my_index',
    func.cast(Item.embedding, HALFVEC(3)).label('embedding'),
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'halfvec_l2_ops'}
)
```

Get the nearest neighbors

```python
order = func.cast(Item.embedding, HALFVEC(3)).l2_distance([3, 1, 2])
session.scalars(select(Item).order_by(order).limit(5))
```

#### Arrays

Add an array column

```python
from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY

class Item(Base):
    embeddings = mapped_column(ARRAY(Vector(3)))
```

And register the types with the underlying driver

For Psycopg 3, use

```python
from pgvector.psycopg import register_vector
from sqlalchemy import event

@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    register_vector(dbapi_connection)
```

For [async connections](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) with Psycopg 3, use

```python
from pgvector.psycopg import register_vector_async
from sqlalchemy import event

@event.listens_for(engine.sync_engine, "connect")
def connect(dbapi_connection, connection_record):
    dbapi_connection.run_async(register_vector_async)
```

For Psycopg 2, use

```python
from pgvector.psycopg2 import register_vector
from sqlalchemy import event

@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    register_vector(dbapi_connection, arrays=True)
```

## SQLModel

Enable the extension

```python
session.exec(text('CREATE EXTENSION IF NOT EXISTS vector'))
```

Add a vector column

```python
from pgvector.sqlalchemy import Vector

class Item(SQLModel, table=True):
    embedding: Any = Field(sa_type=Vector(3))
```

Also supports `HALFVEC`, `BIT`, and `SPARSEVEC`

Insert a vector

```python
item = Item(embedding=[1, 2, 3])
session.add(item)
session.commit()
```

Get the nearest neighbors to a vector

```python
session.exec(select(Item).order_by(Item.embedding.l2_distance([3, 1, 2])).limit(5))
```

Also supports `max_inner_product`, `cosine_distance`, `l1_distance`, `hamming_distance`, and `jaccard_distance`

Get the distance

```python
session.exec(select(Item.embedding.l2_distance([3, 1, 2])))
```

Get items within a certain distance

```python
session.exec(select(Item).filter(Item.embedding.l2_distance([3, 1, 2]) < 5))
```

Average vectors

```python
from pgvector.sqlalchemy import avg

session.exec(select(avg(Item.embedding))).first()
```

Also supports `sum`

Add an approximate index

```python
from sqlmodel import Index

index = Index(
    'my_index',
    Item.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'vector_l2_ops'}
)
# or
index = Index(
    'my_index',
    Item.embedding,
    postgresql_using='ivfflat',
    postgresql_with={'lists': 100},
    postgresql_ops={'embedding': 'vector_l2_ops'}
)

index.create(engine)
```

Use `vector_ip_ops` for inner product and `vector_cosine_ops` for cosine distance

## Psycopg 3

Enable the extension

```python
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
```

Register the types with your connection

```python
from pgvector.psycopg import register_vector

register_vector(conn)
```

For [connection pools](https://www.psycopg.org/psycopg3/docs/advanced/pool.html), use

```python
def configure(conn):
    register_vector(conn)

pool = ConnectionPool(..., configure=configure)
```

For [async connections](https://www.psycopg.org/psycopg3/docs/advanced/async.html), use

```python
from pgvector.psycopg import register_vector_async

await register_vector_async(conn)
```

Create a table

```python
conn.execute('CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3))')
```

Insert a vector

```python
embedding = np.array([1, 2, 3])
conn.execute('INSERT INTO items (embedding) VALUES (%s)', (embedding,))
```

Get the nearest neighbors to a vector

```python
conn.execute('SELECT * FROM items ORDER BY embedding <-> %s LIMIT 5', (embedding,)).fetchall()
```

Add an approximate index

```python
conn.execute('CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)')
# or
conn.execute('CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)')
```

Use `vector_ip_ops` for inner product and `vector_cosine_ops` for cosine distance

## Psycopg 2

Enable the extension

```python
cur = conn.cursor()
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
```

Register the types with your connection or cursor

```python
from pgvector.psycopg2 import register_vector

register_vector(conn)
```

Create a table

```python
cur.execute('CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3))')
```

Insert a vector

```python
embedding = np.array([1, 2, 3])
cur.execute('INSERT INTO items (embedding) VALUES (%s)', (embedding,))
```

Get the nearest neighbors to a vector

```python
cur.execute('SELECT * FROM items ORDER BY embedding <-> %s LIMIT 5', (embedding,))
cur.fetchall()
```

Add an approximate index

```python
cur.execute('CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)')
# or
cur.execute('CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)')
```

Use `vector_ip_ops` for inner product and `vector_cosine_ops` for cosine distance

## asyncpg

Enable the extension

```python
await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
```

Register the types with your connection

```python
from pgvector.asyncpg import register_vector

await register_vector(conn)
```

or your pool

```python
async def init(conn):
    await register_vector(conn)

pool = await asyncpg.create_pool(..., init=init)
```

Create a table

```python
await conn.execute('CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3))')
```

Insert a vector

```python
embedding = np.array([1, 2, 3])
await conn.execute('INSERT INTO items (embedding) VALUES ($1)', embedding)
```

Get the nearest neighbors to a vector

```python
await conn.fetch('SELECT * FROM items ORDER BY embedding <-> $1 LIMIT 5', embedding)
```

Add an approximate index

```python
await conn.execute('CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)')
# or
await conn.execute('CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)')
```

Use `vector_ip_ops` for inner product and `vector_cosine_ops` for cosine distance

## pg8000

Enable the extension

```python
conn.run('CREATE EXTENSION IF NOT EXISTS vector')
```

Register the types with your connection

```python
from pgvector.pg8000 import register_vector

register_vector(conn)
```

Create a table

```python
conn.run('CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3))')
```

Insert a vector

```python
embedding = np.array([1, 2, 3])
conn.run('INSERT INTO items (embedding) VALUES (:embedding)', embedding=embedding)
```

Get the nearest neighbors to a vector

```python
conn.run('SELECT * FROM items ORDER BY embedding <-> :embedding LIMIT 5', embedding=embedding)
```

Add an approximate index

```python
conn.run('CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)')
# or
conn.run('CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)')
```

Use `vector_ip_ops` for inner product and `vector_cosine_ops` for cosine distance

## Peewee

Add a vector column

```python
from pgvector.peewee import VectorField

class Item(BaseModel):
    embedding = VectorField(dimensions=3)
```

Also supports `HalfVectorField`, `FixedBitField`, and `SparseVectorField`

Insert a vector

```python
item = Item.create(embedding=[1, 2, 3])
```

Get the nearest neighbors to a vector

```python
Item.select().order_by(Item.embedding.l2_distance([3, 1, 2])).limit(5)
```

Also supports `max_inner_product`, `cosine_distance`, `l1_distance`, `hamming_distance`, and `jaccard_distance`

Get the distance

```python
Item.select(Item.embedding.l2_distance([3, 1, 2]).alias('distance'))
```

Get items within a certain distance

```python
Item.select().where(Item.embedding.l2_distance([3, 1, 2]) < 5)
```

Average vectors

```python
from peewee import fn

Item.select(fn.avg(Item.embedding).coerce(True)).scalar()
```

Also supports `sum`

Add an approximate index

```python
Item.add_index('embedding vector_l2_ops', using='hnsw')
```

Use `vector_ip_ops` for inner product and `vector_cosine_ops` for cosine distance

## Reference

### Half Vectors

Create a half vector from a list

```python
vec = HalfVector([1, 2, 3])
```

Or a NumPy array

```python
vec = HalfVector(np.array([1, 2, 3]))
```

Get a list

```python
lst = vec.to_list()
```

Get a NumPy array

```python
arr = vec.to_numpy()
```

### Sparse Vectors

Create a sparse vector from a list

```python
vec = SparseVector([1, 0, 2, 0, 3, 0])
```

Or a NumPy array

```python
vec = SparseVector(np.array([1, 0, 2, 0, 3, 0]))
```

Or a SciPy sparse array

```python
arr = coo_array(([1, 2, 3], ([0, 2, 4],)), shape=(6,))
vec = SparseVector(arr)
```

Or a dictionary of non-zero elements

```python
vec = SparseVector({0: 1, 2: 2, 4: 3}, 6)
```

Note: Indices start at 0

Get the number of dimensions

```python
dim = vec.dimensions()
```

Get the indices of non-zero elements

```python
indices = vec.indices()
```

Get the values of non-zero elements

```python
values = vec.values()
```

Get a list

```python
lst = vec.to_list()
```

Get a NumPy array

```python
arr = vec.to_numpy()
```

Get a SciPy sparse array

```python
arr = vec.to_coo()
```

## History

View the [changelog](https://github.com/pgvector/pgvector-python/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/pgvector/pgvector-python/issues)
- Fix bugs and [submit pull requests](https://github.com/pgvector/pgvector-python/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/pgvector/pgvector-python.git
cd pgvector-python
pip install -r requirements.txt
createdb pgvector_python_test
pytest
```

To run an example:

```sh
cd examples/loading
pip install -r requirements.txt
createdb pgvector_example
python3 example.py
```
