from django import forms
from django.db.models import Field
from .. import Vector


# https://docs.djangoproject.com/en/5.0/howto/custom-model-fields/
class VectorField(Field):
    description = 'Vector'
    empty_strings_allowed = False

    def __init__(self, *args, dimensions=None, **kwargs):
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.dimensions is not None:
            kwargs['dimensions'] = self.dimensions
        return name, path, args, kwargs

    def db_type(self, connection):
        if self.dimensions is None:
            return 'vector'
        return 'vector(%d)' % self.dimensions

    def from_db_value(self, value, expression, connection):
        return Vector._from_db(value)

    def to_python(self, value):
        if value is None or isinstance(value, Vector):
            return value
        elif isinstance(value, str):
            return Vector._from_db(value)
        else:
            return Vector(value)

    def get_prep_value(self, value):
        return Vector._to_db(value)

    def value_to_string(self, obj):
        return self.get_prep_value(self.value_from_object(obj))

    def formfield(self, **kwargs):
        return super().formfield(form_class=VectorFormField, **kwargs)


class VectorWidget(forms.TextInput):
    def format_value(self, value):
        if isinstance(value, Vector):
            value = value.to_list()
        return super().format_value(value)


class VectorFormField(forms.CharField):
    widget = VectorWidget

    def to_python(self, value):
        if isinstance(value, str) and value == '':
            return None
        return super().to_python(value)
