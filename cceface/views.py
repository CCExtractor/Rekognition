import datetime
import json
import os
import cv2
import math
# import uuid
import time
import skvideo.io
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


def predict_image(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return "No file part"

        file_ext = str((request.FILES['file'].name)).split('.')[-1]
        filename = id_generator() + '.' + file_ext
        # filename = str(uuid.uuid4())+'.' +file_ext
        file_path = os.path.join(MEDIA_ROOT, 'images/' + filename)
        handle_uploaded_file(request.FILES['file'], file_path)
        file = request.FILES['file']
        # filename = file.name
        try:
            file_form = InputImage(title=filename, imagefile=file.read())
            file_form.save()
        except Exception as e:
            print(e)

        if filename == "":
            return "No selected file"

        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            img = imread(fname=file, mode='RGB')
            if (img.shape[2] == 4):
                img = img[..., :3]

            try:
                all_faces, all_bb = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                all_face_dict = {}
                # print(len(all_faces))
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
                    # print(all_face_dict)
                    # final_result = {"Faces":all_face_dict}
                    return render(request, 'predict_result.html', {'Faces': all_face_dict, 'imagefile': filename})
                else:
                    render(request, 'predict_result.html', {'Faces': '0', 'imagefile': filename})
            except Exception:
                render(request, 'predict_result.html', {'Faces': '0', 'imagefile': filename})
        else:
            render(request, 'predict_result.html', {'Faces': 'Bad request', 'imagefile': filename})
    else:
        return "POST HTTP method required"


def face_vid(request):
    start = time.clock()
    print('hola')
    try:
        if request.method == 'POST':

            file_ext = str((request.FILES['file'].name)).split('.')[-1]
            filename = id_generator() + '.' + file_ext
            # filename = str(uuid.uuid4())+'.' +file_ext
            file_path = os.path.join(MEDIA_ROOT, 'videos/' + filename)
            handle_uploaded_file(request.FILES['file'], file_path)
            file = request.FILES['file']
            # filename = file.name
            try:
                file_form = InputVideo(title=filename, videofile=file.read())
                file_form.save()
            except Exception as e:
                print(e)

        else:
            pass
            # form = VideoForm()
        videofile = file_path

        # print('bhola', filename)
        cap = cv2.VideoCapture(videofile)
        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frame / fps
        sim_cal = int(math.ceil(fps / 10))
        gap = (total_duration / total_frame) * sim_cal * 3 * 1000
        # print(' fps : ',fps,' | tf : ' ,total_frame,' | dur: ', total_duration, ' | frame_hop :' ,sim_cal, ' |  frame gap in ms : ',gap)
        calc_timestamps = [0.0]
        count = 0
        cele = {}
        ids = []
        embedding_dict = load_embeddings(embeddings_path)
        while(cap.isOpened()):
            count = count + 1
            frame_exists, curr_frame = cap.read()

            if (frame_exists):
                if count % sim_cal == 0:
                    timestamps = (cap.get(cv2.CAP_PROP_POS_MSEC))
                    # print(count)
                    try:
                        all_faces, all_bb = get_face(img=curr_frame, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                        # all_face_dict = {}
                        # print(len(all_faces))
                        if all_faces is not None:
                            cele_id = []

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

                                    if(str(id_name) not in ids):
                                        ids.append(str(id_name))
                                        cele[str(id_name)] = []
                                    cele_id.append(id_name)
                                    cele[str(id_name)].append(timestamps)
                            # print(count,cele_id)
                        else:
                            print('No face in the image')
                    except Exception as e:
                        print(e)
                        continue
                    calc_timestamps.append(calc_timestamps[-1] + 1000 / fps)
            else:
                break
        output_dur = time_dura(cele, gap)
        # print(output_dur)
        cap.release()
        print('time taken by facevid : ',time.clock() - start)

        return render(request, 'facevid_result.html', {'dura': output_dur, 'videofile': filename})
    except Exception as e:
        raise(e)


def newface_vid(request):
    start = time.clock()
    print('hola')
    try:
        if request.method == 'POST':

            file_ext = str((request.FILES['file'].name)).split('.')[-1]
            filename = id_generator() + '.' + file_ext
            # filename = str(uuid.uuid4())+'.' +file_ext
            file_path = os.path.join(MEDIA_ROOT, 'videos/' + filename)
            handle_uploaded_file(request.FILES['file'], file_path)
            file = request.FILES['file']
            # filename = file.name
            try:
                file_form = InputVideo(title=filename, videofile=file.read())
                file_form.save()
            except Exception as e:
                print(e)
            # form = VideoForm()
            videofile = file_path

            # print('bhola', filename)
            metadata = skvideo.io.ffprobe(videofile)
            str_fps = metadata["video"]['@avg_frame_rate'].split('/')
            fps = float(float(str_fps[0]) / float(str_fps[1]))

            timestamps = [(float(1) / fps)]
            total_frame=float(metadata["video"]["@nb_frames"])
            total_duration=float(metadata["video"]["@duration"])

            sim_cal=int(math.ceil(fps / 10))
            gap=(total_duration / total_frame) * sim_cal * 3 * 1000

            print(' fps : ',fps,' | tf : ' ,total_frame,' | dur: ', total_duration, ' | frame_hop :' ,sim_cal, ' |  frame gap in ms : ',gap)
            count=0
            cele={}
            ids=[]
            embedding_dict=load_embeddings(embeddings_path)

            videogen=skvideo.io.vreader(videofile)
            for curr_frame in (videogen):
                count=count + 1
                if count % sim_cal == 0:
                    timestamps = (float(count) / fps)*1000 # multiplying to get the timestamps in milliseconds
                    # print(count)
                    try:
                        all_faces, all_bb=get_face(img=curr_frame, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                        if all_faces is not None:
                            cele_id=[]
                            for img, bb in zip(all_faces, all_bb):
                                embedding=embed_image(
                                    img=img, session=facenet_persistent_session,
                                    images_placeholder=images_placeholder, embeddings=embeddings,
                                    phase_train_placeholder=phase_train_placeholder,
                                    image_size=image_size
                                )

                                if embedding_dict:
                                    identity=identify_face(embedding=embedding, embedding_dict=embedding_dict)
                                    identity=identity.split('/')
                                    id_name=identity[len(identity) - 1]

                                    if(str(id_name) not in ids):
                                        ids.append(str(id_name))
                                        cele[str(id_name)]=[]
                                    cele_id.append(id_name)
                                    cele[str(id_name)].append(timestamps)
                            print(count,cele_id)
                        else:
                            print('No face in the image')
                    except Exception as e:
                        print(e)
                        pass
            output_dur=time_dura(cele, gap)
            print('time taken by newfacevid : ',time.clock() - start)
            return render(request, 'facevid_result.html', {'dura': output_dur, 'videofile': filename})
    except Exception as e:
        raise(e)

class API_name_list(views.APIView):
    def get(self, request):
        result=[]
        for fname in os.listdir(embeddings_path):
            fname=os.path.splitext(os.path.join(embeddings_path, fname))
            if fname[1] == ".npy":
                fname=fname[0].split('/')
                result.append(fname[-1])
        return Response(result, status=status.HTTP_200_OK)


class API_feedback(views.APIView):
    def post(self, request):
        if request.method == 'POST':
            if 'file' not in request.FILES:
                return Response(str('No image found'), status=status.HTTP_400_BAD_REQUEST)
            file=request.FILES['file']
            filename='img_' + id_generator() + '_' + file.name

            if file and allowed_file(filename=filename, allowed_set=allowed_set):
                filename=secure_filename(filename=filename)
                img=imread(fname=file, mode='RGB')
                try:
                    img, tmp=get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                    # print(tmp)
                    if img is not None:
                        embedding=embed_image(
                            img=img, session=facenet_persistent_session,
                            images_placeholder=images_placeholder, embeddings=embeddings,
                            phase_train_placeholder=phase_train_placeholder,
                            image_size=image_size
                        )
                        save_image(img=img, filename=filename, upload_path=upload_path)
                        filename=remove_file_extension(filename=filename)
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
    start_time=datetime.datetime.now()
    elapsed_time=datetime.datetime.now() - start_time
    elapsed_time_ms=(elapsed_time.days * 86400000) + (elapsed_time.seconds * 1000) + (elapsed_time.microseconds / 1000)
    return_data={
        "error": "0",
        "message": "Successful",
        "restime": elapsed_time_ms
    }
    return HttpResponse(json.dumps(return_data), content_type='application/json; charset=utf-8')
