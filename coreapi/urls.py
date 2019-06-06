# from rest_framework.decorators import api_view
from django.urls import path
from . import views

urlpatterns = [

    path('image/', views.API_predict_image.as_view(), name='api_image'),
    path('video/', views.API_predict_video.as_view(), name='api_video'),

]