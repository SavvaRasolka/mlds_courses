from django import forms
from django.core.validators import FileExtensionValidator, MaxLengthValidator, MinLengthValidator


class GenerateRequestForm(forms.Form):
    image = forms.ImageField(
        label='Upload image',
        help_text='Supported formats: PNG, JPG, JPEG.',
    )

    text = forms.CharField(
        label='Text prompt',
        help_text='Describe video to generate.',
    )
