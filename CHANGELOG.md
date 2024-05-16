## 0.2.6 (unreleased)

- Added support for `halfvec` and `sparsevec` types to Psycopg 3
- Added support for `halfvec` and `sparsevec` types to asyncpg
- Added support for `halfvec` type to Peewee
- Added `L1Distance` for Django
- Added `l1_distance` for SQLAlchemy, SQLModel, and Peewee

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
