from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, text, Column, Integer
from sqlalchemy.orm import declarative_base, Session
from surprise import Dataset, SVD

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
    factors = Column(Vector(20))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = SVD(n_factors=20, biased=False)
algo.fit(trainset)

users = [dict(id=trainset.to_raw_uid(i), factors=algo.pu[i]) for i in trainset.all_users()]
items = [dict(id=trainset.to_raw_iid(i), factors=algo.qi[i]) for i in trainset.all_items()]

session = Session(engine)
session.bulk_insert_mappings(User, users)
session.bulk_insert_mappings(Item, items)
session.commit()

user = session.query(User).get(1)
items = session.query(Item).order_by(Item.factors.max_inner_product(user.factors)).limit(5).all()
print('user-based recs:', [item.id for item in items])

item = session.query(Item).get(50)
items = session.query(Item).filter(Item.id != item.id).order_by(Item.factors.cosine_distance(item.factors)).limit(5).all()
print('item-based recs:', [item.id for item in items])
