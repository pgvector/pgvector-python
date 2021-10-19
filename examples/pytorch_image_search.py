import matplotlib.pyplot as plt
from pgvector.psycopg import register_vector
import psycopg
import tempfile
import torch
import torchvision
from tqdm import tqdm

seed = True


# establish connection
conn = psycopg.connect(dbname='pgvector_example')
conn.autocommit = True
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)


# load images
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = torchvision.datasets.CIFAR10(root=tempfile.gettempdir(), train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000)


# load pretrained model
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()


def generate_embeddings(inputs):
    return model(inputs).detach().numpy()


# generate, save, and index embeddings
if seed:
    conn.execute('DROP TABLE IF EXISTS image')
    conn.execute('CREATE TABLE image (id bigserial primary key, embedding vector(512))')

    for data in tqdm(dataloader):
        embeddings = generate_embeddings(data[0])

        sql = 'INSERT INTO image (embedding) VALUES ' + ','.join(['(%s)' for _ in embeddings])
        params = [embedding for embedding in embeddings]
        conn.execute(sql, params)

    conn.execute('CREATE INDEX ON image USING ivfflat (embedding vector_cosine_ops)')


def show_images(dataset_images):
    grid = torchvision.utils.make_grid(dataset_images)
    img = (grid / 2 + 0.5).permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.waitforbuttonpress()


# load 5 random unseen images
queryset = torchvision.datasets.CIFAR10(root=tempfile.gettempdir(), train=False, download=True, transform=transform)
queryloader = torch.utils.data.DataLoader(queryset, batch_size=5, shuffle=True)
images = next(iter(queryloader))[0]


# generate and query embeddings
embeddings = generate_embeddings(images)
for image, embedding in zip(images, embeddings):
    result = conn.execute('SELECT id FROM image ORDER BY embedding <=> %s LIMIT 15', (embedding,)).fetchall()
    show_images([image] + [dataset[row[0] - 1][0] for row in result])
