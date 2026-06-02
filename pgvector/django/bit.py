from django import forms
from django.db.models import Field
from typing import Any


# https://docs.djangoproject.com/en/5.0/howto/custom-model-fields/
class BitField(Field):
    description = 'Bit string'

    def __init__(self, *args: Any, length: int | None = None, **kwargs: Any) -> None:
        self.length = length
        super().__init__(*args, **kwargs)

    def deconstruct(self) -> tuple[Any, Any, Any, Any]:
        name, path, args, kwargs = super().deconstruct()
        if self.length is not None:
            kwargs['length'] = self.length
        return name, path, args, kwargs

    def db_type(self, connection: Any) -> str:
        if self.length is None:
            return 'bit'
        return 'bit(%d)' % self.length

    def formfield(self, form_class: Any = None, choices_form_class: Any = None, **kwargs: Any) -> forms.Field:
        return super().formfield(
            form_class=BitFormField if form_class is None else form_class,
            choices_form_class=choices_form_class,
            **kwargs
        )


class BitFormField(forms.CharField):
    def to_python(self, value: Any) -> Any:
        if isinstance(value, str) and value == '':
            return None
        return super().to_python(value)
