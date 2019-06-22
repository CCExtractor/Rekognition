from django.db import models
from django.utils import timezone
import uuid
from Rekognition.settings import MEDIA_ROOT
# Create your models here.


class InputVideo(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=80)
    created_on = models.DateTimeField(default=timezone.now, blank=True)

    def __str__(self):
        return self.id


class InputImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=80)
    created_on = models.DateTimeField(default=timezone.now, blank=True)

    def __str__(self):
        return self.id


class InputEmbed(models.Model):
    unid = uuid.uuid4()
    id = models.UUIDField(primary_key=True, default=unid, editable=False)
    title = models.CharField(max_length=80)
    fileurl = models.CharField(max_length=100, default=MEDIA_ROOT + "/face/" + str(unid) + '.jpg', editable=False)
    created_on = models.DateTimeField(default=timezone.now, blank=True)

    def __str__(self):
        return self.id
