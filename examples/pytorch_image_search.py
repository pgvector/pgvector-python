import matplotlib.pyplot as plt
import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, text, Column, Integer
from sqlalchemy.orm import declarative_base, Session
import tempfile
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

seed = True


# establish connection
engine = create_engine('postgresql+psycopg2://localhost/pgvector_example', future=True)
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()

session = Session(engine)
Base = declarative_base()


# define model
class Image(Base):
    __tablename__ = 'image'

    id = Column(Integer, primary_key=True)
    embedding = Column(Vector(512))


# load images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = torchvision.datasets.CIFAR10(root=tempfile.gettempdir(), train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000)


# load pretrained model
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()


def generate_embeddings(inputs):
    return model(inputs).detach().numpy()


# generate and save embeddings
if seed:
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    for data in tqdm(dataloader):
        inputs, labels = data
        embeddings = generate_embeddings(inputs)
        images = [dict(embedding=embeddings[i]) for i in range(embeddings.shape[0])]

        session.bulk_insert_mappings(Image, images)
        session.commit()


def show_images(dataset_images):
    grid = torchvision.utils.make_grid(dataset_images)
    img = (grid / 2 + 0.5).permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.waitforbuttonpress()


# load 5 random unseen images
queryset = torchvision.datasets.CIFAR10(root=tempfile.gettempdir(), train=False, download=True, transform=transform)
queryloader = torch.utils.data.DataLoader(queryset, batch_size=5, shuffle=True)
images, labels = next(iter(queryloader))


# generate embeddings and query
embeddings = generate_embeddings(images)
for i, embedding in enumerate(embeddings):
    result = session.query(Image).order_by(Image.embedding.cosine_distance(embedding)).limit(15).all()
    show_images([images[i]] + [dataset[image.id - 1][0] for image in result])
