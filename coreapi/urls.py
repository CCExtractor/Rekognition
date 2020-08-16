# from rest_framework.decorators import api_view
from django.urls import path
from . import views

urlpatterns = [
    path('image/', views.ImageFr.as_view(), name='image_api'),
    path('old_video/', views.VideoFr.as_view(), name='video_api'),
    path('embed/', views.EMBEDDING.as_view(), name='embed_api'),
    path('video/', views.AsyncVideoFr.as_view(), name='celery_test_api'),
    path('feedback/', views.FeedbackFeature.as_view(), name='feedback_api'),
    path('nsfw/', views.NsfwRecognise.as_view(), name='nsfw'),
    path('ytstream/', views.StreamVideoFr.as_view(), name='youtube_process'),
    path('simface/', views.SimilarFace.as_view(), name='similar_face'),
    path('objects/', views.ObjectDetect.as_view(), name='object_detect'),
    path('scenetext/', views.SceneText.as_view(), name='scene_text'),
    path('scenedetect/', views.SceneDetect.as_view(), name='scene_detect'),
    path('objectsvideo/', views.ObjectDetectVideo.as_view(), name='object_detect_video'),
    path('scenetextvideo/', views.SceneTextVideo.as_view(), name='scene_text_video'),
    path('nsfwvideo/', views.NsfwVideo.as_view(), name='nsfw_video'),
    path('scenevideo/', views.SceneVideo.as_view(), name='scene_video'),

]
