import implicit
from implicit.datasets.movielens import get_movielens
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, text, Column, Integer, String
from sqlalchemy.orm import declarative_base, Session

engine = create_engine('postgresql+psycopg2://localhost/pgvector_example', future=True)
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()

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


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

titles, ratings = get_movielens('100k')
model = implicit.als.AlternatingLeastSquares(factors=20)
model.fit(ratings)

users = [dict(id=i, factors=factors) for i, factors in enumerate(model.user_factors)]
items = [dict(id=i, title=titles[i].decode('utf-8'), factors=factors) for i, factors in enumerate(model.item_factors)]

session = Session(engine)
session.bulk_insert_mappings(User, users)
session.bulk_insert_mappings(Item, items)
session.commit()

user = session.query(User).get(1)
items = session.query(Item).order_by(Item.factors.max_inner_product(user.factors)).limit(5).all()
print('user-based recs:', [item.title for item in items])

item = session.query(Item).filter(Item.title == 'Star Wars (1977)').first()
items = session.query(Item).filter(Item.id != item.id).order_by(Item.factors.cosine_distance(item.factors)).limit(5).all()
print('item-based recs:', [item.title for item in items])
