from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pgvector',
    version='0.1.8',
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
        'pgvector.psycopg',
        'pgvector.psycopg2',
        'pgvector.sqlalchemy',
        'pgvector.utils'
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy'
    ],
    zip_safe=False
)
