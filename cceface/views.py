import datetime
import json
import os
import cv2
import math
import tensorflow as tf
from skimage.io import imread
from django.shortcuts import render
from cceface.facenet_lib.src.align import detect_face
from werkzeug.utils import secure_filename
from django.http import HttpResponse
from Poorface.settings import BASE_DIR, MEDIA_ROOT
from cceface.utils import (load_model, get_face, embed_image, save_embedding, load_embeddings,
                           identify_face, allowed_file, remove_file_extension, save_image, time_dura, handle_uploaded_file, id_generator)
# from .forms import VideoForm, ImageForm

from rest_framework import views
from rest_framework import status
from rest_framework.response import Response

# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
upload_path = os.path.join(BASE_DIR, 'cceface/uploads')
embeddings_path = os.path.join(BASE_DIR, 'cceface/embeddings')
allowed_set = set(['png', 'jpg', 'jpeg'])
model_path = BASE_DIR + '/cceface/model/2017/20170512-110547.pb'
facenet_model = load_model(model_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
image_size = 160
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
facenet_persistent_session = tf.Session(graph=facenet_model, config=config)
pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)

# @api_view(['POST','GET'])


def get_image(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return "No file part"
        file = request.FILES['file']
        filename = 'img_' + id_generator() + '_' + file.name

        if filename == "":
            return "No Selected file"

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

                    return render(request, "upload_result.html", {'status': "Image uploaded and embedded successfully!"})
                else:
                    return render(request, "upload_result.html", {'status': "Humein khed hai ,tasveer upload nai ho pa saka!"})
            except Exception as e:
                print(e)
    else:
        return "POST HTTP method required!"


# @api_view(['POST','GET'])
def predict_image(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return "No file part"

        filename = 'img_' + id_generator() + '_' + str(request.FILES['file'].name)
        file_path = os.path.join(MEDIA_ROOT, 'images/' + filename)
        handle_uploaded_file(request.FILES['file'], file_path)
        file = request.FILES['file']
        # filename = file.name
        if filename == "":
            return "No selected file"

        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            img = imread(fname=file, mode='RGB')
            try:
                img, bb = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                # print(bb)

                if img is not None:
                    embedding = embed_image(
                        img=img, session=facenet_persistent_session,
                        images_placeholder=images_placeholder, embeddings=embeddings,
                        phase_train_placeholder=phase_train_placeholder,
                        image_size=image_size
                    )
                    embedding_dict = load_embeddings()
                    if embedding_dict:
                        identity = identify_face(embedding=embedding, embedding_dict=embedding_dict)
                        identity = identity.split('/')
                        # print(bb)
                        bounding_box = {"top": bb[1], "bottom": bb[3], "left": bb[0], "right": bb[2]}
                        return render(request, 'predict_result.html', {'bb': bounding_box, 'identity': identity[len(identity) - 1], 'imagefile': filename})
                    else:
                        return render(request,
                                      'predict_result.html')
                else:
                    return render(request,
                                  'predict_result.html'
                                  )
            except Exception as e:
                print(e)
    else:
        return "POST HTTP method required"


def face_vid(request):
    print('hola')
    try:
        if request.method == 'POST':
            filename = 'vid_' + id_generator() + '_' + str(request.FILES['file'].name)
            file_path = os.path.join(MEDIA_ROOT, 'videos/' + filename)
            # form = VideoForm(request.POST, request.FILES)
            handle_uploaded_file(request.FILES['file'], file_path)

        else:
            pass
            # form = VideoForm()
        videofile = file_path

        print('bhola', filename)
        cap = cv2.VideoCapture(videofile)
        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frame / fps
        sim_cal = int(math.ceil(fps / 10))
        gap = (total_duration / total_frame) * sim_cal * 3 * 1000
        # print(fps,total_frame,total_duration,sim_cal,gap)
        calc_timestamps = [0.0]
        count = 0
        cele = {}
        ids = []
        embedding_dict = load_embeddings()
        while(cap.isOpened()):
            count = count + 1
            frame_exists, curr_frame = cap.read()

            if (frame_exists):
                if count % sim_cal == 0:
                    timestamps = (cap.get(cv2.CAP_PROP_POS_MSEC))
                    print(count)
                    try:
                        img, bb = get_face(img=curr_frame, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                        if img is not None:
                            embedding = embed_image(
                                img=img, session=facenet_persistent_session,
                                images_placeholder=images_placeholder, embeddings=embeddings,
                                phase_train_placeholder=phase_train_placeholder,
                                image_size=image_size
                            )

                            if embedding_dict:
                                identity = identify_face(embedding=embedding, embedding_dict=embedding_dict)
                                identity = identity.split('/')
                                identity = identity[len(identity) - 1]
                                # print(identity,timestamps)

                                if(str(identity) not in ids):
                                    ids.append(str(identity))
                                    cele[str(identity)] = []
                                cele[str(identity)].append(timestamps)

                        else:
                            print('No face in the image')
                    except Exception as e:
                        print(e)
                        continue
                    calc_timestamps.append(calc_timestamps[-1] + 1000 / fps)
            else:
                break
        output_dur = time_dura(cele, gap)
        cap.release()

        return render(request, 'facevid_result.html', {'dura': output_dur, 'videofile': filename})
    except Exception as e:
        raise(e)


class API_predict_image(views.APIView):
    def post(self, request):
        if request.method == 'POST':
            if 'file' not in request.FILES:
                return Response(str('err'), status=status.HTTP_400_BAD_REQUEST)

            filename = 'img_' + id_generator() + '_' + str(request.FILES['file'].name)
            file_path = os.path.join(MEDIA_ROOT, 'images/' + filename)
            # form = ImageForm(request.POST, request.FILES)
            handle_uploaded_file(request.FILES['file'], file_path)
            file = request.FILES['file']
            # filename = file.name
            if filename == "":
                return Response(str('err'), status=status.HTTP_400_BAD_REQUEST)

            if file and allowed_file(filename=filename, allowed_set=allowed_set):
                img = imread(fname=file, mode='RGB')
                try:
                    img, bb = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                    # print(bb)

                    if img is not None:
                        embedding = embed_image(
                            img=img, session=facenet_persistent_session,
                            images_placeholder=images_placeholder, embeddings=embeddings,
                            phase_train_placeholder=phase_train_placeholder,
                            image_size=image_size
                        )
                        embedding_dict = load_embeddings()
                        if embedding_dict:
                            identity = identify_face(embedding=embedding, embedding_dict=embedding_dict)
                            identity = identity.split('/')
                            # print(bb)
                            bounding_box = {"top": bb[1], "bottom": bb[3], "left": bb[0], "right": bb[2]}
                            predictions = {'identity': identity[len(identity) - 1], 'Bounding Box': bounding_box}
                            return Response(predictions, status=status.HTTP_200_OK)
                        else:
                            return Response(str('error'), status=status.HTTP_400_BAD_REQUEST)
                    else:
                        return Response(str('error'), status=status.HTTP_400_BAD_REQUEST)

                except Exception as e:
                    return Response(str(e), status=status.HTTP_400_BAD_REQUEST)
                    # print(e)
        else:
            return Response(str('err'), status=status.HTTP_400_BAD_REQUEST)


class API_predict_video(views.APIView):
    def post(self, request):
        try:
            if request.method == 'POST':
                filename = 'vid_' + id_generator() + '_' + str(request.FILES['file'].name)
                file_path = os.path.join(MEDIA_ROOT, 'videos/' + filename)
                # form = VideoForm(request.POST, request.FILES)
                handle_uploaded_file(request.FILES['file'], file_path)

            else:
                # form = VideoForm()
                pass
            videofile = file_path
            # print('bhola',filename)
            cap = cv2.VideoCapture(videofile)
            fps = cap.get(cv2.CAP_PROP_FPS)
            timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = total_frame / fps
            sim_cal = int(math.ceil(fps / 10))
            gap = (total_duration / total_frame) * sim_cal * 3 * 1000
            # print(fps,total_frame,total_duration,sim_cal,gap)
            calc_timestamps = [0.0]
            count = 0
            cele = {}
            ids = []
            embedding_dict = load_embeddings()
            while(cap.isOpened()):
                count = count + 1
                frame_exists, curr_frame = cap.read()

                if (frame_exists):
                    if count % sim_cal == 0:
                        timestamps = (cap.get(cv2.CAP_PROP_POS_MSEC))
                        print(count)
                        try:
                            img, bb = get_face(img=curr_frame, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                            if img is not None:
                                embedding = embed_image(
                                    img=img, session=facenet_persistent_session,
                                    images_placeholder=images_placeholder, embeddings=embeddings,
                                    phase_train_placeholder=phase_train_placeholder,
                                    image_size=image_size
                                )

                                if embedding_dict:
                                    identity = identify_face(embedding=embedding, embedding_dict=embedding_dict)
                                    identity = identity.split('/')
                                    identity = identity[len(identity) - 1]
                                    # print(identity,timestamps)

                                    if(str(identity) not in ids):
                                        ids.append(str(identity))
                                        cele[str(identity)] = []
                                    cele[str(identity)].append(timestamps)

                            else:
                                pass
                                # print('No face in the image')
                        except Exception as e:
                            print(e)
                            continue
                        calc_timestamps.append(calc_timestamps[-1] + 1000 / fps)
                else:
                    break
            output_dur = time_dura(cele, gap)
            cap.release()
            return Response(output_dur, status=status.HTTP_200_OK)
        except Exception as e:
            print(e)


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
