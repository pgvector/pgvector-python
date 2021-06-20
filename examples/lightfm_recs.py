from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, Column, Float, Integer, String
from sqlalchemy.orm import declarative_base, Session

engine = create_engine("postgresql+psycopg2://localhost/pgvector_test", future=True)

Base = declarative_base()


class User(Base):
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)
    factors = Column(Vector(20))


class Item(Base):
    __tablename__ = 'item'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    factors = Column(Vector(20))
    bias = Column(Float)


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

user = session.query(User).get(1)
# subtract item bias for negative inner product
items = session.query(Item).order_by(Item.factors.max_inner_product(user.factors) - Item.bias).limit(5).all()
print('user-based recs:', [item.title for item in items])

item = session.query(Item).filter(Item.title == 'Star Wars (1977)').first()
items = session.query(Item).filter(Item.id != item.id).order_by(Item.factors.cosine_distance(item.factors)).limit(5).all()
print('item-based recs:', [item.title for item in items])
