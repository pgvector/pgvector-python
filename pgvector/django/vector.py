from django import forms
from django.core import exceptions
from django.db.models import Field
from django.utils.functional import cached_property
from typing import Any
from .. import Vector


# https://docs.djangoproject.com/en/6.0/howto/custom-model-fields/
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
        return Vector._from_db(value)

    def to_python(self, value: Any) -> list[float] | None:
        if isinstance(value, list):
            return value
        try:
            return Vector._from_db(value)
        except ValueError:
            raise exceptions.ValidationError('Enter a valid vector.', code='invalid')

    @cached_property
    def validators(self) -> list[Any]:
        return [*super().validators, self._validate_dimensions]

    def _validate_dimensions(self, value: Any) -> None:
        try:
            dimensions = len(Vector._from_db(Vector._to_db(value)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise exceptions.ValidationError('Enter a valid vector.', code='invalid')
        if self.dimensions is not None and dimensions != self.dimensions:
            raise exceptions.ValidationError(
                'Expected %(expected)d dimensions, not %(actual)d.',
                code='dimensions',
                params={'expected': self.dimensions, 'actual': dimensions}
            )

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
