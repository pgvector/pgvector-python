from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, select, text, Float, Integer, String
from sqlalchemy.orm import declarative_base, mapped_column, Session

engine = create_engine('postgresql+psycopg2://localhost/pgvector_example')
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()

Base = declarative_base()


class User(Base):
    __tablename__ = 'user'

    id = mapped_column(Integer, primary_key=True)
    factors = mapped_column(Vector(20))


class Item(Base):
    __tablename__ = 'item'

    id = mapped_column(Integer, primary_key=True)
    title = mapped_column(String)
    factors = mapped_column(Vector(20))
    bias = mapped_column(Float)


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

data = fetch_movielens(min_rating=5.0)
model = LightFM(loss='warp', no_components=20)
model.fit(data['train'], epochs=30)

user_biases, user_factors = model.get_user_representations()
item_biases, item_factors = model.get_item_representations()

users = [dict(id=i, factors=factors) for i, factors in enumerate(user_factors)]
items = [dict(id=i, title=data['item_labels'][i], factors=factors, bias=item_biases[i].item()) for i, factors in enumerate(item_factors)]

session = Session(engine)
session.bulk_insert_mappings(User, users)
session.bulk_insert_mappings(Item, items)
session.commit()

user = session.get(User, 1)
# subtract item bias for negative inner product
items = session.scalars(select(Item).order_by(Item.factors.max_inner_product(user.factors) - Item.bias).limit(5))
print('user-based recs:', [item.title for item in items])

# broken due to https://github.com/lyst/lightfm/issues/682
item = session.scalars(select(Item).filter(Item.title == 'Star Wars (1977)')).first()
items = session.scalars(select(Item).filter(Item.id != item.id).order_by(Item.factors.cosine_distance(item.factors)).limit(5))
print('item-based recs:', [item.title for item in items])
