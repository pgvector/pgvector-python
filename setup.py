from setuptools import setup

setup(
    name='pgvector',
    version='0.1.3',
    description='pgvector support for Python',
    url='https://github.com/ankane/pgvector-python',
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
