from rest_framework import serializers
from .models import InputImage, InputVideo, InputEmbed, NameSuggested


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = InputImage
        fields = '__all__'


class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = InputVideo
        fields = '__all__'


class EmbedSerializer(serializers.ModelSerializer):
    class Meta:
        model = InputEmbed
        fields = '__all__'

class NameSuggestedSerializer(serializers.ModelSerializer):
    class Meta:
        model = NameSuggested
        fields = '__all__'
