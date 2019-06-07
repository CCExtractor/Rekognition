import datetime
import json
import os
from skimage.io import imread
from django.shortcuts import render
from werkzeug.utils import secure_filename
from django.http import HttpResponse
from corelib.facenet.utils import (get_face, embed_image, save_embedding, allowed_file, remove_file_extension, save_image, id_generator)
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from corelib.constant import (pnet, rnet, onet, facenet_persistent_session, phase_train_placeholder,
                              embeddings, images_placeholder, image_size, allowed_set, embeddings_path, upload_path)


def get_image(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return "No file part"
        file = request.FILES['file']
        # filename = 'img_' + id_generator() + '_' + file.name

        filename = request.FILES['file'].name

        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            filename = secure_filename(filename=filename)
            img = imread(fname=file, mode='RGB')
            try:
                img, tmp = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                print(len(img))
                # print(tmp)
                if img is not None:

                    embedding = embed_image(
                        img=img[0], session=facenet_persistent_session,
                        images_placeholder=images_placeholder, embeddings=embeddings,
                        phase_train_placeholder=phase_train_placeholder,
                        image_size=image_size
                    )
                    save_image(img=img[0], filename=filename, upload_path=upload_path)
                    filename = remove_file_extension(filename=filename)
                    save_embedding(embedding=embedding, filename=filename, embeddings_path=embeddings_path)

                    return render(request, "upload_result.html", {'status': "Image uploaded and embedded successfully!"})
                else:
                    return render(request, "upload_result.html", {'status': "Humein khed hai ,tasveer upload nai ho pa saka!"})
            except Exception:
                return render(request, "upload_result.html", {'status': "Humein khed hai ,tasveer upload nai ho pa saka!"})
    else:
        return "POST HTTP method required!"


class API_name_list(views.APIView):
    def get(self, request):
        result = []
        for fname in os.listdir(embeddings_path):
            fname = os.path.splitext(os.path.join(embeddings_path, fname))
            if fname[1] == ".npy":
                fname = fname[0].split('/')
                result.append(fname[-1])
        return Response(result, status=status.HTTP_200_OK)


class API_feedback(views.APIView):
    def post(self, request):
        if request.method == 'POST':
            if 'file' not in request.FILES:
                return Response(str('No image found'), status=status.HTTP_400_BAD_REQUEST)
            file = request.FILES['file']
            filename = 'img_' + id_generator() + '_' + file.name

            if file and allowed_file(filename=filename, allowed_set=allowed_set):
                filename = secure_filename(filename=filename)
                img = imread(fname=file, mode='RGB')
                try:
                    img, tmp = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                    # print(tmp)
                    if img is not None:
                        embedding = embed_image(
                            img=img, session=facenet_persistent_session,
                            images_placeholder=images_placeholder, embeddings=embeddings,
                            phase_train_placeholder=phase_train_placeholder,
                            image_size=image_size
                        )
                        save_image(img=img, filename=filename, upload_path=upload_path)
                        filename = remove_file_extension(filename=filename)
                        save_embedding(embedding=embedding, filename=filename, embeddings_path=embeddings_path)

                        return Response(str(filename + ' has been added !'), status=status.HTTP_200_OK)
                    else:
                        return Response(str('unsuccessful'), status=status.HTTP_400_BAD_REQUEST)
                except Exception:
                    return Response(str('unsuccessful'), status=status.HTTP_400_BAD_REQUEST)


# @api_view(['POST','GET'])
def index_page(request):
    return render(request, "index.html")

# @api_view(['POST','GET'])


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
