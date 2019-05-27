from django.db import models
# Create your models here.


class InputVideo(models.Model):
    video = models.FileField(upload_to='videos/')  # upload_to='videos', null=True, verbose_name=""


class InputImage(models.Model):
    image = models.FileField(upload_to='images/')  # upload_to='videos', null=True, verbose_name=""
