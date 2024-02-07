from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pgvector',
    version='0.2.5',
    description='pgvector support for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pgvector/pgvector-python',
    author='Andrew Kane',
    author_email='andrew@ankane.org',
    license='MIT',
    packages=[
        'pgvector.asyncpg',
        'pgvector.django',
        'pgvector.peewee',
        'pgvector.psycopg',
        'pgvector.psycopg2',
        'pgvector.sqlalchemy',
        'pgvector.utils'
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy'
    ],
    zip_safe=False
)
