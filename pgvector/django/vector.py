from django import forms
from django.db.models import Field
import numpy as np
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

    def from_db_value(self, value: Any, expression: Any, connection: Any) -> np.ndarray | None:
        return Vector._from_db(value)

    def to_python(self, value: Any) -> np.ndarray | None:
        if isinstance(value, list):
            return np.array(value, dtype=np.float32)
        return Vector._from_db(value)

    def get_prep_value(self, value: Any) -> str | None:
        return Vector._to_db(value)

    def value_to_string(self, obj: Any) -> str | None:
        return self.get_prep_value(self.value_from_object(obj))

    def validate(self, value: Any, model_instance: Any) -> None:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        super().validate(value, model_instance)

    def run_validators(self, value: Any) -> None:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        super().run_validators(value)

    def formfield(self, form_class: Any = None, choices_form_class: Any = None, **kwargs: Any) -> forms.Field:
        return super().formfield(
            form_class=VectorFormField if form_class is None else form_class,
            choices_form_class=choices_form_class,
            **kwargs
        )


class VectorWidget(forms.TextInput):
    def format_value(self, value: Any) -> str | None:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return super().format_value(value)


class VectorFormField(forms.CharField):
    widget = VectorWidget

    def has_changed(self, initial: Any, data: Any) -> bool:
        if isinstance(initial, np.ndarray):
            initial = initial.tolist()
        return super().has_changed(initial, data)

    def to_python(self, value: Any) -> Any:
        if isinstance(value, str) and value == '':
            return None
        return super().to_python(value)
