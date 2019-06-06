# from rest_framework.decorators import api_view
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_page, name='index_page'),
    path('upload', views.get_image, name='get_image'),
    path('predictImage', views.predict_image, name='predict_image'),
    path('predict', views.predict_page, name='predict_page'),
    path('facevid', views.facevid_page, name='facevid_page'),
    path('facevid_result', views.face_vid, name='face_vid'),
    path('apidoc', views.api_page, name='api_page'),

]
