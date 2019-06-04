from django.db import models
import uuid
# Create your models here.


class InputVideo(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=80)
    videofile = models.FileField(upload_to='videos/')  # upload_to='videos', null=True, verbose_name=""
    created_on = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.id


class InputImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=80)
    imagefile = models.FileField(upload_to='images/')  # upload_to='videos', null=True, verbose_name=""
    created_on = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.id
