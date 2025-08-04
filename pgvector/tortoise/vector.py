from .. import Vector
from typing import Optional, Union, List, Any
import numpy as np
from tortoise import Model, fields


class VectorField(fields.Field):
    """
    Vector field for Tortoise ORM
    """

    def __init__(self, dimensions: Optional[int] = None, **kwargs):
        self.dimensions = dimensions
        super().__init__(**kwargs)

    @property
    def SQL_TYPE(self) -> str:  # type: ignore
        return f"VECTOR({self.dimensions})"

    def to_db_value(
        self,
        value: Union[np.ndarray, List[float], None],
        instance: type[Model] | Model,
    ) -> Optional[str]:
        return Vector._to_db(value)

    def to_python_value(self, value: Any) -> Optional[np.ndarray]:
        if isinstance(value, list):
            return np.array([float(x) for x in value], dtype=np.float32)
        return Vector._from_db(value)

    def validate(self, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        super().validate(value)
