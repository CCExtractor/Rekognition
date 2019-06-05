import datetime
import json
import os
import cv2
import math
import uuid
from skimage.io import imread
from django.shortcuts import render
from werkzeug.utils import secure_filename
from django.http import HttpResponse
from Rekognition.settings import MEDIA_ROOT
from corelib.facenet.utils import (get_face, embed_image, save_embedding, load_embeddings,
                                   identify_face, allowed_file, remove_file_extension, save_image, time_dura, handle_uploaded_file, id_generator)

from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from corelib.constant import (pnet, rnet, onet, facenet_persistent_session, phase_train_placeholder,
                              embeddings, images_placeholder, image_size, allowed_set, embeddings_path, upload_path)

from .models import InputImage, InputVideo
from .forms import ImageForm, VideoForm

from rest_framework.decorators import api_view



class API_predict_image(views.APIView):
    def post(self, request):
        if request.method == 'POST':
            if 'file' not in request.FILES:
                return Response(str('err'), status=status.HTTP_400_BAD_REQUEST)

            file_ext = str((request.FILES['file'].name)).split('.')[-1]
            filename = id_generator() + '.' + file_ext
            file_path = os.path.join(MEDIA_ROOT, 'images/' + filename)
            handle_uploaded_file(request.FILES['file'], file_path)
            file = request.FILES['file']
            try:
                file_form = InputImage(title=filename, imagefile=file.read())
                file_form.save()
            except Exception as e:

                return Response(str('err'), status=status.HTTP_400_BAD_REQUEST)

            if file and allowed_file(filename=filename, allowed_set=allowed_set):
                img = imread(fname=file, mode='RGB')
                if (img.shape[2] == 4):
                    img = img[..., :3]
                try:
                    all_faces, all_bb = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                    all_face_dict = {}
                    if all_faces is not None:
                        embedding_dict = load_embeddings(embeddings_path)
                        for img, bb in zip(all_faces, all_bb):
                            embedding = embed_image(
                                img=img, session=facenet_persistent_session,
                                images_placeholder=images_placeholder, embeddings=embeddings,
                                phase_train_placeholder=phase_train_placeholder,
                                image_size=image_size
                            )

                            if embedding_dict:
                                identity = identify_face(embedding=embedding, embedding_dict=embedding_dict)
                                identity = identity.split('/')
                                id_name = identity[len(identity) - 1]
                                bounding_box = {"top": bb[1], "bottom": bb[3], "left": bb[0], "right": bb[2]}
                                all_face_dict[id_name] = {"Bounding Boxes": bounding_box}
                        final_result = {"Faces": all_face_dict}
                        return Response(final_result, status=status.HTTP_200_OK)
                    else:
                        return Response(str('error'), status=status.HTTP_400_BAD_REQUEST)

                except Exception as e:
                    return Response(str(e), status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(str('err'), status=status.HTTP_400_BAD_REQUEST)
