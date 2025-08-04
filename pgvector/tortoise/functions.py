from pypika_tortoise import SqlContext

from typing import Any


from tortoise.expressions import F
from tortoise.functions import Function
from pypika_tortoise.terms import Function as PypikaFunction


class BaseVectorFunction(PypikaFunction):
    arg_joiner = "<->"

    def __init__(self, field: F, value: Any):
        super().__init__("", field, value)
        self.emb = value
        self.field = field

    def get_function_sql(self, ctx: SqlContext) -> str:
        return "({field} {arg_joiner} '{vector}'::vector) AS ".format(
            field=self.field, vector=self.emb, arg_joiner=self.arg_joiner
        )


class CosineDistanceAnnotation(Function):
    class CosineDistanceFunction(BaseVectorFunction):
        arg_joiner = "<=>"

    database_func = CosineDistanceFunction


class L2DistanceAnnotation(Function):
    class L2DistanceFunction(BaseVectorFunction):
        arg_joiner = "<->"

    database_func = L2DistanceFunction


class MaxInnerProductAnnotation(Function):
    class MaxInnerProductFunction(BaseVectorFunction):
        arg_joiner = "<#>"

    database_func = MaxInnerProductFunction


class L1DistanceAnnotation(Function):
    class L1DistanceFunction(BaseVectorFunction):
        arg_joiner = "<+>"

    database_func = L1DistanceFunction
