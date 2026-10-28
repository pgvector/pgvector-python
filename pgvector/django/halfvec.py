from django import forms
from django.core import exceptions
from django.db.models import Field
from django.utils.functional import cached_property
from typing import Any
from .. import HalfVector


# https://docs.djangoproject.com/en/6.0/howto/custom-model-fields/
class HalfVectorField(Field):
    description = 'Half vector'
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
            return 'halfvec'
        return 'halfvec(%d)' % self.dimensions

    def from_db_value(self, value: Any, expression: Any, connection: Any) -> list[float] | None:
        return HalfVector._from_db(value)

    def to_python(self, value: Any) -> list[float] | None:
        if isinstance(value, list):
            return value
        try:
            return HalfVector._from_db(value)
        except ValueError:
            raise exceptions.ValidationError('Enter a valid half vector.', code='invalid')

    @cached_property
    def validators(self) -> list[Any]:
        return [*super().validators, self._validate_dimensions]

    def _validate_dimensions(self, value: Any) -> None:
        try:
            vector = HalfVector._from_db(HalfVector._to_db(value))
        except (TypeError, ValueError):
            raise exceptions.ValidationError('Enter a valid half vector.', code='invalid')
        if self.dimensions is not None and vector is not None and len(vector) != self.dimensions:
            raise exceptions.ValidationError(
                'Expected %(expected)d dimensions, not %(actual)d.',
                code='dimensions',
                params={'expected': self.dimensions, 'actual': len(vector)}
            )

    def get_prep_value(self, value: Any) -> str | None:
        return HalfVector._to_db(value)

    def value_to_string(self, obj: Any) -> str:
        value = self.get_prep_value(self.value_from_object(obj))
        return '' if value is None else value

    def formfield(self, form_class: Any = None, choices_form_class: Any = None, **kwargs: Any) -> forms.Field:
        return super().formfield(
            form_class=HalfVectorFormField if form_class is None else form_class,
            choices_form_class=choices_form_class,
            **kwargs
        )


class HalfVectorFormField(forms.CharField):
    def to_python(self, value: Any) -> Any:
        if isinstance(value, str) and value == '':
            return None
        return super().to_python(value)
