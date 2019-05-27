from django import forms


class VideoForm(forms.Form):
    video = forms.FileField(label="Upload a video File")


class ImageForm(forms.Form):
    image = forms.FileField(label="Upload a image File")
