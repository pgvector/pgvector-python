"""
A simple sqlmodel vector demo via pgvector.

For mac, if depdency missing or error, try `pip install pgvector-binary`
"""

from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlmodel import Column, Field, Session, SQLModel, create_engine, select


class Item(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    embedding: List[float] = Field(sa_column=Column(Vector(3)))

sqlite_url = f"postgresql://testuser:testuser@localhost:5432/testdb"

engine = create_engine(sqlite_url, echo=False)

SQLModel.metadata.create_all(engine)

with Session(engine) as session:
    item = Item(embedding=[1, 2, 3])
    session.add(item)
    session.commit()

    res = session.exec(
        select(Item).order_by(Item.embedding.l2_distance([3, 1, 2])).limit(5)
    )

    for i in res:
        print(i)
