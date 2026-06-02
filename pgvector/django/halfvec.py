from django import forms
from django.db.models import Field
from typing import Any
from .. import HalfVector


# https://docs.djangoproject.com/en/5.0/howto/custom-model-fields/
class HalfVectorField(Field):
    description = 'Half vector'
    empty_strings_allowed = False

    def __init__(self, *args: Any, dimensions: int | None = None, **kwargs: Any) -> None:
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    def deconstruct(self) -> tuple:
        name, path, args, kwargs = super().deconstruct()
        if self.dimensions is not None:
            kwargs['dimensions'] = self.dimensions
        return name, path, args, kwargs

    def db_type(self, connection: Any) -> str:
        if self.dimensions is None:
            return 'halfvec'
        return 'halfvec(%d)' % self.dimensions

    def from_db_value(self, value: Any, expression: Any, connection: Any) -> HalfVector | None:
        return HalfVector._from_db(value)

    def to_python(self, value: Any) -> HalfVector | None:
        if value is None or isinstance(value, HalfVector):
            return value
        elif isinstance(value, str):
            return HalfVector._from_db(value)
        else:
            return HalfVector(value)

    def get_prep_value(self, value: Any) -> str | None:
        return HalfVector._to_db(value)

    def value_to_string(self, obj: Any) -> str | None:
        return self.get_prep_value(self.value_from_object(obj))

    def formfield(self, **kwargs) -> forms.Field:  # type: ignore
        return super().formfield(form_class=HalfVectorFormField, **kwargs)


class HalfVectorWidget(forms.TextInput):
    def format_value(self, value: Any) -> str | None:
        if isinstance(value, HalfVector):
            value = value.to_list()
        return super().format_value(value)


class HalfVectorFormField(forms.CharField):
    widget = HalfVectorWidget

    def to_python(self, value: Any) -> Any:
        if isinstance(value, str) and value == '':
            return None
        return super().to_python(value)
