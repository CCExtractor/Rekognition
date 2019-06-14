# from rest_framework.decorators import api_view
from django.urls import path
from . import views
from coreapi.views import ImageWebUI, VideoWebUI

urlpatterns = [
    path('', views.index_page, name='index_page'),
    path('predictImage', ImageWebUI, name='predict_image'),
    path('predict', views.predict_page, name='predict_page'),
    path('facevid', views.facevid_page, name='facevid_page'),
    path('facevid_result', VideoWebUI, name='face_vid'),
    path('apidoc', views.api_page, name='api_page'),

]
