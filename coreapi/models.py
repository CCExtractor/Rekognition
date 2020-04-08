from django.db import models
from django.utils import timezone
import uuid
# Create your models here.


class InputVideo(models.Model):
    """     Class to represent the input videos in the database

    Workflow
            *   id : primary key for input videos
            *   title : name of input videos
            *   isProcessesd : set true for processed videos and false for others
            *   created_on : date and time of the input video

    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=80)
    isProcessed = models.BooleanField(default=False)
    created_on = models.DateTimeField(default=timezone.now, blank=True)

    def __str__(self):
        return self.title


class InputImage(models.Model):
    """     Class to represent the input images in the database

    Workflow
            *   id : primary key for input image
            *   title : name of input image
            *   isProcessesd : set true for processed image and false for others
            *   created_on : date and time of the input image

    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=80,)
    isProcessed = models.BooleanField(default=False)
    created_on = models.DateTimeField(default=timezone.now, blank=True)

    def __str__(self):
        return self.title


class InputEmbed(models.Model):
    """     Class to represent the embeddings in the database

    Workflow
            *   id : primary key for embeddings
            *   title : name of embeddings file
            *   fileurl : path of the saved embedding
            *   created_on : date and time of the embeddings

    """
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
    """     Class to represent the suggested name in the database for the feedback feature

    Workflow
            *   suggestedName : Suggested Name for the embedding
            *   upvote : number of upvotes for the suggested name
            *   downvote : number of downvotes for the suggested name
            *   feedback : foreign key that connects NameSuggested to InputEmbed

    """
    suggestedName = models.CharField(max_length=80)
    upvote = models.IntegerField(default=0)
    downvote = models.IntegerField(default=0)
    feedback = models.ForeignKey(InputEmbed, on_delete=models.CASCADE)

    def __str__(self):
        return self.suggestedName


class SimilarFaceInImage(models.Model):
    """     Class to represent the Similar Faces that have been found in image in the database

    Workflow
            *   id : primary key for image of the Similar Face
            *   title : name of image of the Similar Face
            *   similarwith : name of the image file which has the face that it is similar to 
            *   created_on : date and time of the image of the Similar Face

    """
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
