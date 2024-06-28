from datasets import load_dataset
import matplotlib.pyplot as plt
from pgvector.psycopg import register_vector, Bit
import psycopg
from imagehash import phash


def hash_image(img):
    return ''.join(['1' if v else '0' for v in phash(img).hash.flatten()])


conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS images')
conn.execute('CREATE TABLE images (id bigserial PRIMARY KEY, hash bit(64))')

print('Loading dataset')
dataset = load_dataset('mnist')

print('Generating hashes')
images = [{'hash': hash_image(row['image'])} for row in dataset['train']]

print('Storing hashes')
cur = conn.cursor()
with cur.copy('COPY images (hash) FROM STDIN') as copy:
    for image in images:
        copy.write_row([Bit(image['hash'])])

print('Querying hashes')
results = []
for i in range(5):
    image = dataset['test'][i]['image']
    result = conn.execute('SELECT id FROM images ORDER BY hash <~> %s LIMIT 5', (hash_image(image),)).fetchall()
    nearest_images = [dataset['train'][row[0] - 1]['image'] for row in result]
    results.append([image] + nearest_images)

print('Showing results (first column is query image)')
fig, axs = plt.subplots(len(results), len(results[0]))
for i, result in enumerate(results):
    for j, image in enumerate(result):
        ax = axs[i, j]
        ax.imshow(image)
        ax.set_axis_off()
plt.show(block=True)
