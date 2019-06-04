from django.db import models
import uuid
# Create your models here.


class InputVideo(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=80)
    video = models.FileField(upload_to='videos/')  # upload_to='videos', null=True, verbose_name=""

    def __str__(self):
        return self.title


class InputImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=80)
    image = models.FileField(upload_to='images/')  # upload_to='videos', null=True, verbose_name=""

    def __str__(self):
        return self.title
