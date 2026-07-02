from django import forms
from django.db.models import Field
from typing import Any
from .. import Vector


# https://docs.djangoproject.com/en/5.0/howto/custom-model-fields/
class VectorField(Field):
    description = 'Vector'
    empty_strings_allowed = False

    def __init__(self, *args: Any, dimensions: int | None = None, **kwargs: Any) -> None:
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    def deconstruct(self) -> tuple[Any, Any, Any, Any]:
        name, path, args, kwargs = super().deconstruct()
        if self.dimensions is not None:
            kwargs['dimensions'] = self.dimensions
        return name, path, args, kwargs

    def db_type(self, connection: Any) -> str:
        if self.dimensions is None:
            return 'vector'
        return 'vector(%d)' % self.dimensions

    def from_db_value(self, value: Any, expression: Any, connection: Any) -> list[float] | None:
        value = Vector._from_db(value)
        return None if value is None else value.to_list()

    def to_python(self, value: Any) -> list[float] | None:
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            value = Vector.from_text(value)
        elif not isinstance(value, Vector):
            value = Vector(value)
        return value.to_list()  # type: ignore

    def get_prep_value(self, value: Any) -> str | None:
        return Vector._to_db(value)

    def value_to_string(self, obj: Any) -> str:
        value = self.get_prep_value(self.value_from_object(obj))
        return '' if value is None else value

    def formfield(self, form_class: Any = None, choices_form_class: Any = None, **kwargs: Any) -> forms.Field:
        return super().formfield(
            form_class=VectorFormField if form_class is None else form_class,
            choices_form_class=choices_form_class,
            **kwargs
        )


class VectorFormField(forms.CharField):
    def to_python(self, value: Any) -> Any:
        if isinstance(value, str) and value == '':
            return None
        return super().to_python(value)
