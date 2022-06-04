from django.shortcuts import render
from rest_framework import views, status
from rest_framework.response import Response
from corelib.facenet.utils import handle_uploaded_file
from Rekognition.settings import MEDIA_ROOT
import os
from corelib.facenet.utils import (getnewuniquefilename)
from corelib.main_api import (facerecogniseinimage, facerecogniseinvideo,
                              createembedding, process_streaming_video,
                              nsfwclassifier, similarface, object_detect,
                              text_detect, object_detect_video, scene_detect,
                              text_detect_video, scene_video, nsfw_video)
from .serializers import (EmbedSerializer, NameSuggestedSerializer,
                          SimilarFaceSerializer, ImageFrSerializers)
from .models import InputEmbed, NameSuggested, SimilarFaceInImage
from logger.logging import RekogntionLogger
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
import asyncio
from threading import Thread
import random
import tracemalloc
import time


logger = RekogntionLogger(name="view")


class SceneText(views.APIView):
    """     To localize and recognise text in an image
    Workflow
            *   if  POST method request is made, then initially a random
                filename is generated and then text_detect method is
                called which process the image and outputs the result
                containing the dictionary of detected text and bounding
                boxes of the text
    Returns:
            *   output dictionary of detected text and bounding
                boxes of the text
    """

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for Scene Text Extraction made")
        filename = getnewuniquefilename(request)
        input_file = request.FILES['file']
        result = text_detect(input_file, filename)
        if "Error" not in result:
            logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
            end = time.time()
            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            tracemalloc.stop()

            return Response(result, status=status.HTTP_200_OK)
        else:
            if (result["Error"] == 'An HTTP error occurred.'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'A Connection error occurred.'):
                return Response(result, status=status.HTTP_503_SERVICE_UNAVALIABLE)
            elif (result["Error"] == 'The request timed out.'):
                return Response(result, status=status.HTTP_408_REQUEST_TIMEOUT)
            elif (result["Error"] == 'Bad URL'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'Text Detection Not Working'):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            elif (result["Error"] == 'The media format of the requested data is not supported by the server'):
                return Response(result, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            elif (result["Error"] == 'A JSON error occurred.'):
                return Response(result, status=status.HTTP_204_NO_CONTENT)
            elif (result["Error"] == 'A proxy error occurred.'):
                return Response(result, status=status.HTTP_407_PROXY_AUTHENTICATION_REQUIRED)
            elif (result["Error"] == 'The header value provided was somehow invalid.'):
                return Response(result, status=status.HTTP_411_LENGTH_REQUIRED)
            elif (result["Error"] == 'The request timed out while trying to connect to the remote server.'):
                return Response(result, status=status.HTTP_504_GATEWAY_TIMEOUT)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)


class SceneTextVideo(views.APIView):
    """     To localize and recognise text in a video
    Workflow
            *   if  POST method request is made, then initially a random
                filename is generated and then text_detect_video method
                is called which process the video and outputs the result
                containing the dictionary of detected text and bounding
                boxes of the text for each frame
    Returns:
            *   output dictionary of detected text and bounding
                boxes of the text for each frame of the video
    """

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for Scene Text Extraction in video made")
        filename = getnewuniquefilename(request)
        input_file = request.FILES['file']
        result = text_detect_video(input_file, filename)
        if "Error" not in result:
            end = time.time()
            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
            tracemalloc.stop()
            return Response(result, status=status.HTTP_200_OK)
        else:
            if (result["Error"] == 'An HTTP error occurred.'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'A Connection error occurred.'):
                return Response(result, status=status.HTTP_503_SERVICE_UNAVALIABLE)
            elif (result["Error"] == 'The request timed out.'):
                return Response(result, status=status.HTTP_408_REQUEST_TIMEOUT)
            elif (result["Error"] == 'Bad URL'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'Text Detect(video) Not Working'):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            elif (result["Error"] == 'The media format of the requested data is not supported by the server'):
                return Response(result, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            elif (result["Error"] == 'A JSON error occurred.'):
                return Response(result, status=status.HTTP_204_NO_CONTENT)
            elif (result["Error"] == 'A proxy error occurred.'):
                return Response(result, status=status.HTTP_407_PROXY_AUTHENTICATION_REQUIRED)
            elif (result["Error"] == 'The header value provided was somehow invalid.'):
                return Response(result, status=status.HTTP_411_LENGTH_REQUIRED)
            elif (result["Error"] == 'The request timed out while trying to connect to the remote server.'):
                return Response(result, status=status.HTTP_504_GATEWAY_TIMEOUT)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)


class NsfwRecognise(views.APIView):
    """     To recognise whether a image is nsfw or not
    Workflow
            *   if  POST method request is made, then initially a random
                filename is generated and then nsfwclassifier method is
                called which process the image and outputs the result
                containing the dictionary of probability of type of content
                in the image
    Returns:
            *   output dictionary of probability content in the image
    """

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for NSFW Classification made")
        filename = getnewuniquefilename(request)
        input_file = request.FILES['file']
        result = nsfwclassifier(input_file, filename)
        if "Error" not in result:
            end = time.time()
            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
            tracemalloc.stop()
            return Response(result, status=status.HTTP_200_OK)
        else:
            if (result["Error"] == 'An HTTP error occurred.'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'A Connection error occurred.'):
                return Response(result, status=status.HTTP_503_SERVICE_UNAVALIABLE)
            elif (result["Error"] == 'The request timed out.'):
                return Response(result, status=status.HTTP_408_REQUEST_TIMEOUT)
            elif (result["Error"] == 'Bad URL'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'NSFW Classification Not Working'):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            elif (result["Error"] == 'The media format of the requested data is not supported by the server'):
                return Response(result, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            elif (result["Error"] == 'A JSON error occurred.'):
                return Response(result, status=status.HTTP_204_NO_CONTENT)
            elif (result["Error"] == 'A proxy error occurred.'):
                return Response(result, status=status.HTTP_407_PROXY_AUTHENTICATION_REQUIRED)
            elif (result["Error"] == 'The header value provided was somehow invalid.'):
                return Response(result, status=status.HTTP_411_LENGTH_REQUIRED)
            elif (result["Error"] == 'The request timed out while trying to connect to the remote server.'):
                return Response(result, status=status.HTTP_504_GATEWAY_TIMEOUT)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)


class NsfwVideo(views.APIView):
    """     To recognise which frames in a video are NSFW
    Workflow
            *   if  POST method request is made, then initially a random
                filename is generated and then nsfw_video method
                is called which process the video and outputs the result
                containing the the dictionary of probability of type of
                scene in the video for each frame
    Returns:
            *   output dictionary of probability content in the each frame
                of the video
    """

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for NSFW Classification in video made")
        filename = getnewuniquefilename(request)
        input_file = request.FILES['file']
        result = nsfw_video(input_file, filename)
        if "Error" not in result:
            end = time.time()
            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
            tracemalloc.stop()
            return Response(result, status=status.HTTP_200_OK)
        else:
            if (result["Error"] == 'An HTTP error occurred.'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'A Connection error occurred.'):
                return Response(result, status=status.HTTP_503_SERVICE_UNAVALIABLE)
            elif (result["Error"] == 'The request timed out.'):
                return Response(result, status=status.HTTP_408_REQUEST_TIMEOUT)
            elif (result["Error"] == 'Bad URL'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'NSFW Classification(video) Not Working'):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            elif (result["Error"] == 'The media format of the requested data is not supported by the server'):
                return Response(result, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            elif (result["Error"] == 'A JSON error occurred.'):
                return Response(result, status=status.HTTP_204_NO_CONTENT)
            elif (result["Error"] == 'A proxy error occurred.'):
                return Response(result, status=status.HTTP_407_PROXY_AUTHENTICATION_REQUIRED)
            elif (result["Error"] == 'The header value provided was somehow invalid.'):
                return Response(result, status=status.HTTP_411_LENGTH_REQUIRED)
            elif (result["Error"] == 'The request timed out while trying to connect to the remote server.'):
                return Response(result, status=status.HTTP_504_GATEWAY_TIMEOUT)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)


class SceneDetect(views.APIView):
    """     To classify scene in an image
    Workflow
            *   if  POST method request is made, then initially a random
                filename is generated and then scene_detect method is
                called which process the image and outputs the result
                containing the dictionary of probability of type of
                scene in the image
    Returns:
            *   output dictionary of detected scenes and probabilities
                of scenes in image
    """

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for Scene Detection made")
        filename = getnewuniquefilename(request)
        input_file = request.FILES['file']
        result = scene_detect(input_file, filename)
        if "Error" not in result:
            end = time.time()
            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
            tracemalloc.stop()
            return Response(result, status=status.HTTP_200_OK)
        else:
            if (result["Error"] == 'An HTTP error occurred.'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'A Connection error occurred.'):
                return Response(result, status=status.HTTP_503_SERVICE_UNAVALIABLE)
            elif (result["Error"] == 'The request timed out.'):
                return Response(result, status=status.HTTP_408_REQUEST_TIMEOUT)
            elif (result["Error"] == 'Bad URL'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'Scene Detect Not Working'):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            elif (result["Error"] == 'The media format of the requested data is not supported by the server'):
                return Response(result, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            elif (result["Error"] == 'A JSON error occurred.'):
                return Response(result, status=status.HTTP_204_NO_CONTENT)
            elif (result["Error"] == 'A proxy error occurred.'):
                return Response(result, status=status.HTTP_407_PROXY_AUTHENTICATION_REQUIRED)
            elif (result["Error"] == 'The header value provided was somehow invalid.'):
                return Response(result, status=status.HTTP_411_LENGTH_REQUIRED)
            elif (result["Error"] == 'The request timed out while trying to connect to the remote server.'):
                return Response(result, status=status.HTTP_504_GATEWAY_TIMEOUT)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)


class SceneVideo(views.APIView):
    """     To classify scenes video
    Workflow
            *   if  POST method request is made, then initially a random
                filename is generated and then scene_video method
                is called which process the video and outputs the result
                containing the the dictionary of probability of type of
                content in the video for each frame
    Returns:
            *   output dictionary of probability scene in the each frame
                of the video
    """

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for Scene Classification in video made")
        filename = getnewuniquefilename(request)
        input_file = request.FILES['file']
        result = scene_video(input_file, filename)
        if "Error" not in result:
            end = time.time()
            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
            tracemalloc.stop()
            return Response(result, status=status.HTTP_200_OK)
        else:
            if (result["Error"] == 'An HTTP error occurred.'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'A Connection error occurred.'):
                return Response(result, status=status.HTTP_503_SERVICE_UNAVALIABLE)
            elif (result["Error"] == 'The request timed out.'):
                return Response(result, status=status.HTTP_408_REQUEST_TIMEOUT)
            elif (result["Error"] == 'Bad URL'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'Scence Video Not Working'):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            elif (result["Error"] == 'The media format of the requested data is not supported by the server'):
                return Response(result, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            elif (result["Error"] == 'A JSON error occurred.'):
                return Response(result, status=status.HTTP_204_NO_CONTENT)
            elif (result["Error"] == 'A proxy error occurred.'):
                return Response(result, status=status.HTTP_407_PROXY_AUTHENTICATION_REQUIRED)
            elif (result["Error"] == 'The header value provided was somehow invalid.'):
                return Response(result, status=status.HTTP_411_LENGTH_REQUIRED)
            elif (result["Error"] == 'The request timed out while trying to connect to the remote server.'):
                return Response(result, status=status.HTTP_504_GATEWAY_TIMEOUT)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)


class ImageFr(views.APIView):
    """     To recognise faces in image
    Workflow\n
            *   if  POST method request is made, then initially a random
                filename is generated and then facerecogniseinimage method
                is called which process the image and outputs the result
                containing all the information about the faces available
                in the image.
    Returns\n
            *   output by facerecogniseinimage
    """
    serializer = ImageFrSerializers

    def get(self, request):
        logger.info(msg="GET Request for Face Reocgnition made")
        serializer = self.serializer()
        return Response(serializer.data)

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for Face Recognition made")
        image_serializer = self.serializer(data=request.data)
        filename = getnewuniquefilename(request)
        input_file = request.FILES['file']
        if image_serializer.is_valid():
            network = image_serializer.data["network"]
            result = facerecogniseinimage(input_file, filename, network)
            if "Error" not in result:
                end = time.time()
                logger.info(msg="Time For Prediction = " + str(int(end - start)))
                result['Time'] = int(end - start)
                result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
                logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
                tracemalloc.stop()
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)

        else:
            logger.error(msg=image_serializer.errors)
            return Response(image_serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)


class VideoFr(views.APIView):
    """     To recognise faces in video
    Workflow
            *   if  POST method request is made, then initially a random
                filename is generated and then facerecogniseinvideo method
                is called which process the video and outputs the result
                containing all the information about the faces available
                in the video.
    Returns:
            *   output by facerecogniseinvideo
    """

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for Face Recognition in Video made")
        filename = getnewuniquefilename(request)
        input_file = request.FILES['file']
        file_path = os.path.join(MEDIA_ROOT, 'videos', filename)
        handle_uploaded_file(input_file, file_path)
        result = facerecogniseinvideo(input_file, filename)
        if "Error" not in result:
            end = time.time()
            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
            tracemalloc.stop()
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)


class EMBEDDING(views.APIView):
    """     To create embedding of faces
    Workflow
            *   if  GET method request is made, all the faceid are returned
            *   if  POST method request is made, then the file is sent to
                createembedding to create the embedding
    Returns:
            *   POST : output whether it was successful or not
            *   GET  : List the data stored in database
    """
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        logger.info(msg="GET Request for generating embeddings made")
        embedlist = InputEmbed.objects.all()
        serializer = EmbedSerializer(embedlist, many=True)
        return Response({'data': serializer.data})

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for generating embeddings made")
        filename = request.FILES['file'].name
        input_file = request.FILES['file']
        result = createembedding(input_file, filename)
        if "Error" not in result:
            end = time.time()
            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
            tracemalloc.stop()
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)


class FeedbackFeature(APIView):
    """     Feedback feature
    Workflow
            *   if GET method request is made, then first all the embeddings
                objects are loaded followed by randomly selecting anyone
                of them.
            *   with the help of id of the randomly selected object,
                an attempt is made to get object available in NameSuggested
                model. If the object is available then it is selected else
                a new object is created in NameSuggested model .
            *   All the objects having the ids are fetched and serialized and
                then passed to reponse the request.
            *   if POST method request is made, then first the received data
                is made mutable so later the embedding object can be included
                in the data.
            *   With the help of id contained in the POST request embedding
                object is fetched and attached to the data followed by
                serializing it , Now here is a catch, How the POST request
                know the id which is present in the database? This is actually
                answered by the GET request. When GET request is made it sends
                a feedback_id which is used to make POST request when ever a
                new name is suggested to the faceid.
            *   So, if there is any action on already available NameSuggested
                object i.e. upvote or downvote then the object is updated in
                the database else a new object is made with the same id having
                upvote = downvote = 0. Here don't mix id and primary key.
                Primary key in this case is different than this id.
    """
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        embedlist = InputEmbed.objects.all()
        randomfaceobject = embedlist[random.randrange(len(embedlist))]
        try:
            namesuggestedobject = NameSuggested.objects.get(feedback_id=randomfaceobject.id)
        except NameSuggested.MultipleObjectsReturned:
            namesuggestedobject = NameSuggested.objects.filter(feedback_id=randomfaceobject.id).first()
            logger.warn(msg="Multiple names were returned, first anme has been set.")
        except NameSuggested.DoesNotExist:
            namesuggestedobject = NameSuggested.objects.create(suggested_name=randomfaceobject.title,
                                                               feedback=randomfaceobject)
            namesuggestedobject.save()
            logger.warn(msg="No names were returned, random name has been set.")
        namesuggestedlist = NameSuggested.objects.filter(feedback_id=randomfaceobject.id)
        serializer = NameSuggestedSerializer(namesuggestedlist, many=True)
        result = {'data': serializer.data,
                  'fileurl': randomfaceobject.fileurl}
        return Response(result)

    def post(self, request, *args, **kwargs):
        request.data._mutable = True
        feedbackmodel = InputEmbed.objects.get(id=request.data["feedback_id"])
        request.data["feedback"] = feedbackmodel
        feedback_serializer = NameSuggestedSerializer(data=request.data)
        if feedback_serializer.is_valid():
            try:
                obj = NameSuggested.objects.get(id=request.data["id"])
                obj.upvote = request.data["upvote"]
                obj.downvote = request.data["downvote"]
                obj.save()
            except NameSuggested.DoesNotExist:
                feedback_serializer.save()
                logger.warn(msg="No names were returned, random name has been set.")
            return Response(feedback_serializer.data,
                            status=status.HTTP_201_CREATED)
        else:
            logger.error(msg=feedback_serializer.errors)
            return Response(feedback_serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)


def imagewebui(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            logger.error(msg="file not found")
            return render(request, '404.html')
        else:
            filename = getnewuniquefilename(request)
            result = facerecogniseinimage(request, filename)
            if "Error" not in result:
                return render(request, 'predict_result.html',
                              {'Faces': result, 'imagefile': filename})
            else:
                return render(request, 'predict_result.html',
                              {'Faces': result, 'imagefile': filename})
    else:
        logger.error(msg="GET request made instead of POST")
        return "POST HTTP method required!"


def videowebui(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            logger.error(msg="file not found")
            return render(request, '404.html')
        else:
            filename = getnewuniquefilename(request)
            result = facerecogniseinvideo(request, filename)
            if "Error" not in result:
                return render(request, 'facevid_result.html',
                              {'dura': result, 'videofile': filename})
            else:
                return render(request, 'facevid_result.html',
                              {'dura': result, 'videofile': filename})
    else:
        logger.error(msg="GET request made instead of POST")
        return "POST HTTP method required!"


async def async_helper(request, filename):
    return (facerecogniseinvideo(request.FILES['file'], filename))


def asyncthread(request, filename):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_helper(request, filename))
    loop.close()


class AsyncVideoFr(views.APIView):
    def post(self, request):
        tracemalloc.start()
        start = time.time()
        filename = getnewuniquefilename(request)
        file_path = os.path.join(MEDIA_ROOT, 'videos', filename)
        input_file = request.FILES['file']
        handle_uploaded_file(input_file, file_path)
        thread = ThreadWithReturnValue(target=facerecogniseinvideo, args=(input_file, filename))
        thread.start()
        result = thread.join()
        end = time.time()
        logger.info(msg="Time For Prediction = " + str(int(end - start)))
        result['Time'] = int(end - start)
        result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
        logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
        tracemalloc.stop()
        if "Error" not in result:
            return Response(result, status=status.HTTP_200_OK)
        else:
            if (result["Error"] == 'An HTTP error occurred.'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'A Connection error occurred.'):
                return Response(result, status=status.HTTP_503_SERVICE_UNAVALIABLE)
            elif (result["Error"] == 'The request timed out.'):
                return Response(result, status=status.HTTP_408_REQUEST_TIMEOUT)
            elif (result["Error"] == 'Bad URL'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'Face Recongiton(Video) Not Working'):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            elif (result["Error"] == 'The media format of the requested data is not supported by the server'):
                return Response(result, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            elif (result["Error"] == 'A JSON error occurred.'):
                return Response(result, status=status.HTTP_204_NO_CONTENT)
            elif (result["Error"] == 'A proxy error occurred.'):
                return Response(result, status=status.HTTP_407_PROXY_AUTHENTICATION_REQUIRED)
            elif (result["Error"] == 'The header value provided was somehow invalid.'):
                return Response(result, status=status.HTTP_411_LENGTH_REQUIRED)
            elif (result["Error"] == 'The request timed out while trying to connect to the remote server.'):
                return Response(result, status=status.HTTP_504_GATEWAY_TIMEOUT)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)


class StreamVideoFr(views.APIView):
    """     To recognise faces in YouTube video
    Workflow
            *   youtube embed link is received by reactjs post request then it
                is preprocessed to get the original youtube link and then
                it is passed
    Returns:
            *   output by facerecogniseinvideo
    """

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for Procesing Youtube Videos made")
        streamlink = request.data["StreamLink"]
        videoid = (str(streamlink).split('/')[-1]).split('\"')[0]
        ytlink = str("https://www.youtube.com/watch?v=" + str(videoid))
        result = process_streaming_video(ytlink, (videoid))
        if "Error" not in result:
            end = time.time()
            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
            tracemalloc.stop()
            return Response(result, status=status.HTTP_200_OK)
        else:
            if (result["Error"] == 'An HTTP error occurred.'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'A Connection error occurred.'):
                return Response(result, status=status.HTTP_503_SERVICE_UNAVALIABLE)
            elif (result["Error"] == 'The request timed out.'):
                return Response(result, status=status.HTTP_408_REQUEST_TIMEOUT)
            elif (result["Error"] == 'Bad URL'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'Video Processing Not Working'):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            elif (result["Error"] == 'The media format of the requested data is not supported by the server'):
                return Response(result, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            elif (result["Error"] == 'A JSON error occurred.'):
                return Response(result, status=status.HTTP_204_NO_CONTENT)
            elif (result["Error"] == 'A proxy error occurred.'):
                return Response(result, status=status.HTTP_407_PROXY_AUTHENTICATION_REQUIRED)
            elif (result["Error"] == 'The header value provided was somehow invalid.'):
                return Response(result, status=status.HTTP_411_LENGTH_REQUIRED)
            elif (result["Error"] == 'The request timed out while trying to connect to the remote server.'):
                return Response(result, status=status.HTTP_504_GATEWAY_TIMEOUT)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)


class SimilarFace(views.APIView):
    """     To recognise similar faces in two images
    Workflow
            *   if  POST method request is made, then initially a random
                filename is generated and then similarface method is called
                which process the image and outputs the result containing the
                dictionary of file name and image id of matched face
    Returns:
            *   output by similarface
    """

    def get(self, request, *args, **kwargs):
        logger.info(msg="GET Request for Similar Face Recognition made")
        similarfacelist = SimilarFaceInImage.objects.all()
        serializer = SimilarFaceSerializer(similarfacelist, many=True)
        return Response({'data': serializer.data})

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for Similar Face Recognition made")
        filename = getnewuniquefilename(request)
        reference_img = request.FILES['file']
        compare_img = request.FILES['compareImage']
        result = similarface(reference_img, compare_img, filename)
        if "Error" not in result:
            end = time.time()
            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
            tracemalloc.stop()
            return Response(result, status=status.HTTP_200_OK)
        else:
            if (result["Error"] == 'An HTTP error occurred.'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'A Connection error occurred.'):
                return Response(result, status=status.HTTP_503_SERVICE_UNAVALIABLE)
            elif (result["Error"] == 'The request timed out.'):
                return Response(result, status=status.HTTP_408_REQUEST_TIMEOUT)
            elif (result["Error"] == 'Bad URL'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'Similar Face Not Working'):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            elif (result["Error"] == 'The media format of the requested data is not supported by the server'):
                return Response(result, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            elif (result["Error"] == 'A JSON error occurred.'):
                return Response(result, status=status.HTTP_204_NO_CONTENT)
            elif (result["Error"] == 'A proxy error occurred.'):
                return Response(result, status=status.HTTP_407_PROXY_AUTHENTICATION_REQUIRED)
            elif (result["Error"] == 'The header value provided was somehow invalid.'):
                return Response(result, status=status.HTTP_411_LENGTH_REQUIRED)
            elif (result["Error"] == 'The request timed out while trying to connect to the remote server.'):
                return Response(result, status=status.HTTP_504_GATEWAY_TIMEOUT)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)


class ObjectDetect(views.APIView):
    """     To detect objects in an image
    Workflow
            *   if  POST method request is made, then initially a random
                filename is generated and then object_detect method is
                called which process the image and outputs the result
                containing the dictionary of detected objects, confidence
                scores and bounding box coordinates
    Returns:
            *   output dictionary of detected objects, confidence scores
                and bounding box coordinates
    """

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for Object Detection made")
        filename = getnewuniquefilename(request)
        input_file = request.FILES['file']
        result = object_detect(input_file, filename)
        if "Error" not in result:
            end = time.time()
            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            logger.info(msg="Memory Used = " + str((tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001))
            tracemalloc.stop()
            return Response(result, status=status.HTTP_200_OK)
        else:
            if (result["Error"] == 'An HTTP error occurred.'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'A Connection error occurred.'):
                return Response(result, status=status.HTTP_503_SERVICE_UNAVALIABLE)
            elif (result["Error"] == 'The request timed out.'):
                return Response(result, status=status.HTTP_408_REQUEST_TIMEOUT)
            elif (result["Error"] == 'Bad URL'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'Object Detection Not Working'):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            elif (result["Error"] == 'The media format of the requested data is not supported by the server'):
                return Response(result, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            elif (result["Error"] == 'A JSON error occurred.'):
                return Response(result, status=status.HTTP_204_NO_CONTENT)
            elif (result["Error"] == 'A proxy error occurred.'):
                return Response(result, status=status.HTTP_407_PROXY_AUTHENTICATION_REQUIRED)
            elif (result["Error"] == 'The header value provided was somehow invalid.'):
                return Response(result, status=status.HTTP_411_LENGTH_REQUIRED)
            elif (result["Error"] == 'The request timed out while trying to connect to the remote server.'):
                return Response(result, status=status.HTTP_504_GATEWAY_TIMEOUT)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)


class ObjectDetectVideo(views.APIView):
    """     To detect objects in a video
    Workflow
            *   if  POST method request is made, then initially a random
                filename is generated and then object_detect_video method is
                called which process the image and outputs the result
                containing the dictionary of detected objects, confidence
                scores and bounding box coordinates for each frame
    Returns:
            *   output dictionary of detected objects, confidence scores
                and bounding box coordinates for each frame of the video
    """

    def post(self, request):

        tracemalloc.start()
        start = time.time()
        logger.info(msg="POST Request for Object Detection in video made")
        filename = getnewuniquefilename(request)
        input_file = request.FILES['file']
        result = object_detect_video(input_file, filename)
        if "Error" not in result:
            end = time.time()

            result['Time'] = int(end - start)

            logger.info(msg="Time For Prediction = " + str(int(end - start)))
            result['Time'] = int(end - start)
            result["Memory"] = (tracemalloc.get_traced_memory()[1] - tracemalloc.get_traced_memory()[0]) * 0.001
            logger.info()
            tracemalloc.stop()
            return Response(result, status=status.HTTP_200_OK)
        else:
            if (result["Error"] == 'An HTTP error occurred.'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'A Connection error occurred.'):
                return Response(result, status=status.HTTP_503_SERVICE_UNAVALIABLE)
            elif (result["Error"] == 'The request timed out.'):
                return Response(result, status=status.HTTP_408_REQUEST_TIMEOUT)
            elif (result["Error"] == 'Bad URL'):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)
            elif (result["Error"] == 'Object Detection(Video) Not Working'):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            elif (result["Error"] == 'The media format of the requested data is not supported by the server'):
                return Response(result, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            elif (result["Error"] == 'A JSON error occurred.'):
                return Response(result, status=status.HTTP_204_NO_CONTENT)
            elif (result["Error"] == 'A proxy error occurred.'):
                return Response(result, status=status.HTTP_407_PROXY_AUTHENTICATION_REQUIRED)
            elif (result["Error"] == 'The header value provided was somehow invalid.'):
                return Response(result, status=status.HTTP_411_LENGTH_REQUIRED)
            elif (result["Error"] == 'The request timed out while trying to connect to the remote server.'):
                return Response(result, status=status.HTTP_504_GATEWAY_TIMEOUT)
            else:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return
