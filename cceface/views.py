import datetime
import json
import os
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from corelib.constant import embeddings_path


class API_name_list(views.APIView):
    def get(self, request):
        result = []
        for fname in os.listdir(embeddings_path):
            fname = os.path.splitext(os.path.join(embeddings_path, fname))
            if fname[1] == ".npy":
                fname = fname[0].split('/')
                result.append(fname[-1])
        return Response(result, status=status.HTTP_200_OK)


# @api_view(['POST','GET'])
def index_page(request):
    return render(request, "index.html")


def predict_page(request):
    return render(request, "predict.html")


def api_page(request):
    return render(request, "api_page.html")


def facevid_page(request):
    return render(request, "facevid.html")
# @api_view(['GET'])


def __index__function(request):
    start_time = datetime.datetime.now()
    elapsed_time = datetime.datetime.now() - start_time
    elapsed_time_ms = (elapsed_time.days * 86400000) + (elapsed_time.seconds * 1000) + (elapsed_time.microseconds / 1000)
    return_data = {
        "error": "0",
        "message": "Successful",
        "restime": elapsed_time_ms
    }
    return HttpResponse(json.dumps(return_data), content_type='application/json; charset=utf-8')
