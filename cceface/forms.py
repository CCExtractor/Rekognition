from django.forms import ModelForm
from .models import InputImage, InputVideo


class VideoForm(ModelForm):
    # required_css_class = 'required'

    class Meta:
        model = InputVideo
        fields = '__all__'


class ImageForm(ModelForm):
    # required_css_class = 'required'

    class Meta:
        model = InputImage
        fields = '__all__'
