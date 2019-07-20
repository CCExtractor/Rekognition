from django.shortcuts import render
from rest_framework import views, status
from rest_framework.response import Response
from corelib.facenet.utils import (getNewUniqueFileName)
from .main_api import FaceRecogniseInImage, FaceRecogniseInVideo, createEmbedding
from .serializers import EmbedSerializer, NameSuggestedSerializer
from .models import InputEmbed, NameSuggested
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
import asyncio
from threading import Thread
import random


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


class FeedbackFeature(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        embedList = InputEmbed.objects.all()
        randomFaceObject = embedList[random.randrange(len(embedList))]
        try:
            nameSuggestedObject = NameSuggested.objects.get(feedback_id=randomFaceObject.id)
        except NameSuggested.MultipleObjectsReturned:
            pass
        except NameSuggested.DoesNotExist:
            nameSuggestedObject = NameSuggested.objects.create(suggestedName=randomFaceObject.title, feedback=randomFaceObject)
            nameSuggestedObject.save()

        nameSuggestedList = NameSuggested.objects.filter(feedback_id=randomFaceObject.id)
        serializer = NameSuggestedSerializer(nameSuggestedList, many=True)
        result = {'data': serializer.data, 'fileurl': randomFaceObject.fileurl}
        return Response(result)

    def post(self, request, *args, **kwargs):
        request.data._mutable = True
        # print(request.data)
        feedbackModel = InputEmbed.objects.get(id=request.data["feedback_id"])
        request.data["feedback"] = feedbackModel
        feedback_serializer = NameSuggestedSerializer(data=request.data)
        if feedback_serializer.is_valid():
            try:
                obj = NameSuggested.objects.get(id=request.data["id"])
                obj.upvote = request.data["upvote"]
                obj.downvote = request.data["downvote"]
                obj.save()
            except NameSuggested.DoesNotExist:
                feedback_serializer.save()
            return Response(feedback_serializer.data, status=status.HTTP_201_CREATED)
        else:
            print('error', feedback_serializer.errors)
            return Response(feedback_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


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


async def ASYNC_helper(request, filename):
    return (FaceRecogniseInVideo(request, filename))


def AsyncThread(request, filename):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ASYNC_helper(request, filename))
    loop.close()


class ASYNC_VIDEOFR(views.APIView):
    def post(self, request):
        if request.method == 'POST':
            filename = getNewUniqueFileName(request)
            thread = Thread(target=AsyncThread, args=(request, filename))
            thread.start()
            return Response(str(filename.split('.')[0]), status=status.HTTP_200_OK)
        else:
            Response(str('Bad POST Request'), status=status.HTTP_400_BAD_REQUEST)
