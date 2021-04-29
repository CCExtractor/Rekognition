from django.db import models
from django.utils import timezone
import uuid
from django.conf import settings
if not settings.configured:
    settings.configure()
# Create your models here.


class InputVideo(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=80)
    is_processed = models.BooleanField(default=False)
    created_on = models.DateTimeField(default=timezone.now, blank=True)

    def __str__(self):
        return self.title


class InputImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=80,)
    is_processed = models.BooleanField(default=False)
    created_on = models.DateTimeField(default=timezone.now, blank=True)

    def __str__(self):
        return self.title


class InputEmbed(models.Model):
    id = models.CharField(primary_key=True, editable=False, max_length=50)
    title = models.CharField(max_length=80)
    fileurl = models.CharField(max_length=100, editable=False)
    created_on = models.DateTimeField(default=timezone.now, blank=True)

    def __str__(self):
        return self.title

    def save(self, **kwargs):
        if not self.id:
            self.id = "{}".format(self.fileurl.split('/')[-1].split('.')[0])
        super().save(**kwargs)


class NameSuggested(models.Model):
    suggested_name = models.CharField(max_length=80)
    upvote = models.IntegerField(default=0)
    downvote = models.IntegerField(default=0)
    feedback = models.ForeignKey(InputEmbed, on_delete=models.CASCADE)

    def __str__(self):
        return self.suggested_name


class SimilarFaceInImage(models.Model):
    id = models.CharField(primary_key=True, editable=False, max_length=50)
    title = models.CharField(max_length=80)
    similarwith = models.CharField(max_length=80)
    created_on = models.DateTimeField(default=timezone.now, blank=True)

    def __str__(self):
        return self.title

    def save(self, **kwargs):
        if not self.id:
            self.id = "{}".format(self.title)
        super().save(**kwargs)
