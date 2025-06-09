## 0.4.2 (2025-06-09)

- Add support for `list[float]` returned from db query when using sqlalchemy core

## 0.4.1 (2025-04-26)

- Fixed `SparseVector` constructor for SciPy sparse matrices

## 0.4.0 (2025-03-15)

- Added top-level `pgvector` package
- Added support for pg8000
- Added support for `bytes` to `Bit` constructor
- Changed `globally` option to default to `False` for Psycopg 2
- Changed `arrays` option to default to `True` for Psycopg 2
- Fixed equality for `Vector`, `HalfVector`, `Bit`, and `SparseVector` classes
- Fixed `indices` and `values` methods of `SparseVector` returning tuple instead of list in some cases
- Dropped support for Python < 3.9

## 0.3.6 (2024-10-26)

- Added `arrays` option for Psycopg 2

## 0.3.5 (2024-10-05)

- Added `avg` function with type casting to SQLAlchemy
- Added `globally` option for Psycopg 2

## 0.3.4 (2024-09-26)

- Added `schema` option for asyncpg

## 0.3.3 (2024-09-09)

- Improved support for cursor factories with Psycopg 2

## 0.3.2 (2024-07-17)

- Fixed error with asyncpg and pgvector < 0.7

## 0.3.1 (2024-07-10)

- Fixed error parsing zero sparse vectors
- Fixed error with Psycopg 2 and pgvector < 0.7
- Fixed error message when `vector` type not found with Psycopg 3

## 0.3.0 (2024-06-25)

- Added support for `halfvec`, `bit`, and `sparsevec` types to Django
- Added support for `halfvec`, `bit`, and `sparsevec` types to SQLAlchemy and SQLModel
- Added support for `halfvec` and `sparsevec` types to Psycopg 3
- Added support for `halfvec` and `sparsevec` types to Psycopg 2
- Added support for `halfvec` and `sparsevec` types to asyncpg
- Added support for `halfvec`, `bit`, and `sparsevec` types to Peewee
- Added `L1Distance`, `HammingDistance`, and `JaccardDistance` for Django
- Added `l1_distance`, `hamming_distance`, and `jaccard_distance` for SQLAlchemy and SQLModel
- Added `l1_distance`, `hamming_distance`, and `jaccard_distance` for Peewee

## 0.2.5 (2024-02-07)

- Added literal binds support for SQLAlchemy

## 0.2.4 (2023-11-24)

- Improved reflection with SQLAlchemy

## 0.2.3 (2023-09-25)

- Fixed null values with Django
- Fixed `full_clean` with Django

## 0.2.2 (2023-09-08)

- Added support for Peewee
- Added `HnswIndex` for Django

## 0.2.1 (2023-07-31)

- Fixed form issues with Django

## 0.2.0 (2023-07-23)

- Fixed form validation with Django
- Dropped support for Python < 3.8

## 0.1.8 (2023-05-20)

- Fixed serialization with Django

## 0.1.7 (2023-05-11)

- Added `register_vector_async` for Psycopg 3
- Fixed `set_types` for Psycopg 3

## 0.1.6 (2022-05-22)

- Fixed return type for distance operators with SQLAlchemy

## 0.1.5 (2022-01-14)

- Fixed `operator does not exist` error with Django
- Fixed warning with SQLAlchemy 1.4.28+

## 0.1.4 (2021-10-12)

- Updated Psycopg 3 integration for 3.0 release (no longer experimental)

## 0.1.3 (2021-06-22)

- Added support for asyncpg
- Added experimental support for Psycopg 3

## 0.1.2 (2021-06-13)

- Added Django support

## 0.1.1 (2021-06-12)

- Added `l2_distance`, `max_inner_product`, and `cosine_distance` for SQLAlchemy

## 0.1.0 (2021-06-11)

- First release
