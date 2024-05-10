from django import forms
import numpy as np
from .widgets import VectorWidget


class VectorFormField(forms.CharField):
    widget = VectorWidget

    def has_changed(self, initial, data):
        if isinstance(initial, np.ndarray):
            initial = initial.tolist()
        return super().has_changed(initial, data)

    def to_python(self, value):
        if value in self.empty_values:
            return None
        return super().to_python(value)