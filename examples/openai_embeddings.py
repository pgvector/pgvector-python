import openai
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, select, text, Integer, String, Text
from sqlalchemy.orm import declarative_base, mapped_column, Session

engine = create_engine('postgresql+psycopg://localhost/pgvector_example')
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()

Base = declarative_base()


class Document(Base):
    __tablename__ = 'document'

    id = mapped_column(Integer, primary_key=True)
    content = mapped_column(Text)
    embedding = mapped_column(Vector(1536))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]

embeddings = [v['embedding'] for v in openai.Embedding.create(input=input, model='text-embedding-ada-002')['data']]
documents = [dict(content=input[i], embedding=embedding) for i, embedding in enumerate(embeddings)]

session = Session(engine)
session.bulk_insert_mappings(Document, documents)
session.commit()

doc = session.get(Document, 1)
neighbors = session.scalars(select(Document).filter(Document.id != doc.id).order_by(Document.embedding.max_inner_product(doc.embedding)).limit(5))
for neighbor in neighbors:
    print(neighbor.content)
