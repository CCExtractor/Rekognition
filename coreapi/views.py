from django.shortcuts import render
from rest_framework import views, status
from rest_framework.response import Response
from corelib.facenet.utils import (getNewUniqueFileName,handle_uploaded_file)
from .main_api import FaceRecogniseInImage, FaceRecogniseInVideo, createEmbedding
from .tasks import CFRVideo, addi
import os
from Rekognition.settings import MEDIA_ROOT
from .serializers import EmbedSerializer
from .models import InputEmbed
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser

class IMAGE_FR(views.APIView):
    def post(self, request):
        if request.method == 'POST':
            filename = getNewUniqueFileName(request)
            result = FaceRecogniseInImage(request, filename)
            if 'error' or 'Error' not in result:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(str('error'), status=status.HTTP_400_BAD_REQUEST)
        else:
            Response(str('Bad GET Request'), status=status.HTTP_400_BAD_REQUEST)


class VIDEO_FR(views.APIView):
    def post(self, request):
        if request.method == 'POST':
            filename = getNewUniqueFileName(request)
            result = FaceRecogniseInVideo(request, filename)
            if 'error' or 'Error' not in result:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(str('error'), status=status.HTTP_400_BAD_REQUEST)
        else:
            Response(str('Bad GET Request'), status=status.HTTP_400_BAD_REQUEST)


class CREATE_EMBEDDING(views.APIView):
    def post(self, request):
        if request.method == 'POST':
            filename = request.FILES['file'].name
            result = createEmbedding(request, filename)
            if 'success' in result:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
        else:
            Response(str('Bad GET Request'), status=status.HTTP_400_BAD_REQUEST)


class LIST_AVAILABLE_EMBEDDING_DETAILS(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        EmbedList = InputEmbed.objects.all()
        serializer = EmbedSerializer(EmbedList, many=True)
        return Response({'data': serializer.data})

    def post(self, request, *args, **kwargs):
        pass
        # Images_serializer = ImageSerializer(data=request.data)
        # if Images_serializer.is_valid():
        #     Images_serializer.save()
        #     return Response(Images_serializer.data, status=status.HTTP_201_CREATED)
        # else:
        #     print('error', Images_serializer.errors)
        #     return Response(Images_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def ImageWebUI(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return render(request, '404.html')
        else:
            filename = getNewUniqueFileName(request)
            result = FaceRecogniseInImage(request, filename)
            if 'error' or 'Error' not in result:
                return render(request, 'predict_result.html', {'Faces': result, 'imagefile': filename})
            else:
                return render(request, 'predict_result.html', {'Faces': result, 'imagefile': filename})
    else:
        return "POST HTTP method required!"


def VideoWebUI(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return render(request, '404.html')
        else:
            filename = getNewUniqueFileName(request)
            result = FaceRecogniseInVideo(request, filename)
            if 'error' or 'Error' not in result:
                return render(request, 'facevid_result.html', {'dura': result, 'videofile': filename})
            else:
                return render(request, 'facevid_result.html', {'dura': result, 'videofile': filename})
    else:
        return "POST HTTP method required!"

import time
class Celerytest(views.APIView):
    def post(self, request):
        if request.method == 'POST':
            filename = getNewUniqueFileName(request)
            print("hola it's starting")
            # test = addi.delay(5,9)
            # print(' ############ ',test.id,test.status,test.get())
            file_path = os.path.join(MEDIA_ROOT, 'videos/' + filename)
            handle_uploaded_file(request.FILES['file'], file_path)
            result = CFRVideo.delay(file_path, filename)
            print("hola it's working")
            print("#"*30,result.id,result.status)
            # time.sleep(10)

            # if 'error' or 'Error' not in result:
            return Response('success', status=status.HTTP_200_OK)
            # else:
            #     return Response(str('error'), status=status.HTTP_400_BAD_REQUEST)
        else:
            Response(str('Bad GET Request'), status=status.HTTP_400_BAD_REQUEST)

