# https://docs.sqlalchemy.org/en/20/core/functions.html
# include sum for a consistent API
from sqlalchemy.sql.functions import ReturnTypeFromArgs, sum
from typing import Any


class avg(ReturnTypeFromArgs[Any]):
    inherit_cache = True
    package = 'pgvector'


__all__ = [
    'avg',
    'sum'
]
