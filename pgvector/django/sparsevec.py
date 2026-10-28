from django import forms
from django.core import exceptions
from django.db.models import Field
from django.utils.functional import cached_property
from typing import Any
from .. import SparseVector


# https://docs.djangoproject.com/en/6.0/howto/custom-model-fields/
class SparseVectorField(Field):
    description = 'Sparse vector'
    empty_strings_allowed = False

    def __init__(self, *args: Any, dimensions: int | None = None, **kwargs: Any):
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    def deconstruct(self) -> tuple[Any, Any, Any, Any]:
        name, path, args, kwargs = super().deconstruct()
        if self.dimensions is not None:
            kwargs['dimensions'] = self.dimensions
        return name, path, args, kwargs

    def db_type(self, connection: Any) -> str:
        if self.dimensions is None:
            return 'sparsevec'
        return 'sparsevec(%d)' % self.dimensions

    def from_db_value(self, value: Any, expression: Any, connection: Any) -> SparseVector | None:
        return SparseVector._from_db(value)

    def to_python(self, value: Any) -> SparseVector | None:
        try:
            return SparseVector._from_db(value)
        except ValueError:
            raise exceptions.ValidationError('Enter a valid sparse vector.', code='invalid')

    @cached_property
    def validators(self) -> list[Any]:
        return [*super().validators, self._validate_dimensions]

    def _validate_dimensions(self, value: Any) -> None:
        try:
            vector = SparseVector._from_db(SparseVector._to_db(value))
        except (TypeError, ValueError):
            raise exceptions.ValidationError('Enter a valid sparse vector.', code='invalid')
        if self.dimensions is not None and vector is not None and vector.dimensions() != self.dimensions:
            raise exceptions.ValidationError(
                'Expected %(expected)d dimensions, not %(actual)d.',
                code='dimensions',
                params={'expected': self.dimensions, 'actual': vector.dimensions()}
            )

    def get_prep_value(self, value: Any) -> str | None:
        return SparseVector._to_db(value)

    def value_to_string(self, obj: Any) -> str:
        value = self.get_prep_value(self.value_from_object(obj))
        return '' if value is None else value

    def formfield(self, form_class: Any = None, choices_form_class: Any = None, **kwargs: Any) -> forms.Field:
        return super().formfield(
            form_class=SparseVectorFormField if form_class is None else form_class,
            choices_form_class=choices_form_class,
            **kwargs
        )


class SparseVectorWidget(forms.TextInput):
    def format_value(self, value: Any) -> Any:
        if isinstance(value, SparseVector):
            value = value.to_text()
        return super().format_value(value)


class SparseVectorFormField(forms.CharField):
    widget = SparseVectorWidget

    def to_python(self, value: Any) -> Any:
        if isinstance(value, str) and value == '':
            return None
        return super().to_python(value)
