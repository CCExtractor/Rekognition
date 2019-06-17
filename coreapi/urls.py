# from rest_framework.decorators import api_view
from django.urls import path
from . import views

urlpatterns = [

    path('image/', views.IMAGE_FR.as_view(), name='image_api'),
    path('video/', views.VIDEO_FR.as_view(), name='video_api'),
    path('faceid/', views.LIST_EMBEDDING.as_view(), name='name_api'),
    path('embed/', views.CREATE_EMBEDDING.as_view(), name='embed_api'),

]
