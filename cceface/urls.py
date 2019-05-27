# from rest_framework.decorators import api_view
from django.urls import path
from . import views

urlpatterns = [
    # ex: /polls/
    path('', views.index_page, name='index_page'),
    path('upload', views.get_image, name='get_image'),
    path('predictImage', views.predict_image, name='predict_image'),
    path('predict', views.predict_page, name='predict_page'),
    path('facevid', views.facevid_page, name='facevid_page'),
    path('facevid_result', views.face_vid, name='face_vid'),
    path('api/image', views.API_predict_image.as_view(), name='image'),
    path('api/video', views.API_predict_video.as_view(), name='video'),
    path('api/namelist', views.API_name_list.as_view(), name='namelist'),
    path('api/feedback', views.API_feedback.as_view(), name='feedback'),
    path('api', views.api_page, name='api_page'),

]
# if settings.DEBUG: # new
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
