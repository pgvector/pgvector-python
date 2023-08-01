from django import forms
from .widgets import VectorWidget


class VectorFormField(forms.CharField):
    widget = VectorWidget

    def has_changed(self, initial, data):
        try:
            initial = initial.tolist()
        except AttributeError:
            # initial could be None
            pass
        return super().has_changed(initial, data)
