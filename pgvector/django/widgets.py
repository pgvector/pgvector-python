from django import forms


class VectorWidget(forms.TextInput):
    def format_value(self, value):
        try:
            value = value.tolist()
        except AttributeError:
            # value could be None
            pass
        return super().format_value(value)
