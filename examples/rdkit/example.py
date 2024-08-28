# good resource
# https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints

from pgvector.psycopg import register_vector, Bit
import psycopg
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_fingerprint(molecule):
    fpgen = AllChem.GetMorganGenerator()
    return fpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(molecule))


conn = psycopg.connect(dbname='pgvector_example', autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS molecules')
conn.execute('CREATE TABLE molecules (id text PRIMARY KEY, fingerprint bit(2048))')

molecules = ['Cc1ccccc1', 'Cc1ncccc1', 'c1ccccn1']
for molecule in molecules:
    fingerprint = generate_fingerprint(molecule)
    conn.execute('INSERT INTO molecules (id, fingerprint) VALUES (%s, %s)', (molecule, Bit(fingerprint)))

query_molecule = 'c1ccco1'
query_fingerprint = generate_fingerprint(query_molecule)
result = conn.execute('SELECT id, fingerprint <%%> %s AS distance FROM molecules ORDER BY distance LIMIT 5', (Bit(query_fingerprint),)).fetchall()
for row in result:
    print(row)
