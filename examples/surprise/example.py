from pgvector.sqlalchemy import VECTOR
from sqlalchemy import create_engine, insert, select, text, Integer
from sqlalchemy.orm import declarative_base, mapped_column, Session
from surprise import Dataset, SVD

engine = create_engine('postgresql+psycopg://localhost/pgvector_example')
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()

Base = declarative_base()


class User(Base):
    __tablename__ = 'user'

    id = mapped_column(Integer, primary_key=True)
    factors = mapped_column(VECTOR(20))


class Item(Base):
    __tablename__ = 'item'

    id = mapped_column(Integer, primary_key=True)
    factors = mapped_column(VECTOR(20))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = SVD(n_factors=20, biased=False)
algo.fit(trainset)

users = [dict(id=trainset.to_raw_uid(i), factors=algo.pu[i]) for i in trainset.all_users()]
items = [dict(id=trainset.to_raw_iid(i), factors=algo.qi[i]) for i in trainset.all_items()]

session = Session(engine)
session.execute(insert(User), users)
session.execute(insert(Item), items)

user = session.get(User, 1)
items = session.scalars(select(Item).order_by(Item.factors.max_inner_product(user.factors)).limit(5))
print('user-based recs:', [item.id for item in items])

item = session.get(Item, 50)
items = session.scalars(select(Item).filter(Item.id != item.id).order_by(Item.factors.cosine_distance(item.factors)).limit(5))
print('item-based recs:', [item.id for item in items])
