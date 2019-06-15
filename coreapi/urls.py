# from rest_framework.decorators import api_view
from django.urls import path
from . import views

urlpatterns = [

    path('image/', views.IMAGE_API.as_view(), name='image_api'),
    path('video/', views.VIDEO_API.as_view(), name='video_api'),
    path('embed/', views.EMBEDDING_API.as_view(), name='embed_api'),
    path('celery/', views.Celerytest.as_view(), name='celery_test_api'),

]
