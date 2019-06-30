import os
import math
import uuid
import json
import skvideo.io
import subprocess
import shlex
from skimage.io import imread
from werkzeug.utils import secure_filename
from Rekognition.settings import MEDIA_ROOT
from corelib.facenet.utils import (get_face, embed_image, save_embedding, embedding_dict,
                                   identify_face, allowed_file, time_dura, handle_uploaded_file, save_face)
from corelib.constant import (pnet, rnet, onet, facenet_persistent_session, phase_train_placeholder,
                              embeddings, images_placeholder, image_size, allowed_set, embeddings_path)
from .models import InputImage, InputVideo, InputEmbed


def FaceRecogniseInImage(request, filename):
    file_path = os.path.join(MEDIA_ROOT, 'images/' + filename)
    handle_uploaded_file(request.FILES['file'], file_path)
    file = request.FILES['file']

    if file and allowed_file(filename=filename, allowed_set=allowed_set):
        try:
            file_form = InputImage(title=filename)
            file_form.save()
        except Exception as e:
            return (e)

        img = imread(fname=file, mode='RGB')
        if (img.shape[2] == 4):
            img = img[..., :3]

        try:
            all_faces, all_bb = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
            all_face_dict = {}
            if all_faces is not None:
                for img, bb in zip(all_faces, all_bb):
                    embedding = embed_image(img=img, session=facenet_persistent_session, images_placeholder=images_placeholder, embeddings=embeddings,
                                            phase_train_placeholder=phase_train_placeholder, image_size=image_size)

                    if embedding_dict:
                        id_name = identify_face(embedding=embedding, embedding_dict=embedding_dict)
                        bounding_box = {"top": bb[1], "bottom": bb[3], "left": bb[0], "right": bb[2]}
                        all_face_dict[id_name] = {"Bounding Boxes": bounding_box}
                # try:
                #     with open(os.path.join(MEDIA_ROOT, 'output/image',filename.split('.')[0]+'.json'), 'w') as fp:
                #         json.dump(all_face_dict, fp)
                # except Exception as e:
                #     print(e)
                    # pass
                file_form.isProcessed = True
                file_form.save()
                return all_face_dict
            else:
                return 'error no faces'
        except Exception as e:
            return e
    else:
        return {"Error": 'bad file format'}


def FaceRecogniseInVideo(request, filename):
    file_path = os.path.join(MEDIA_ROOT, 'videos/' + filename)
    handle_uploaded_file(request.FILES['file'], file_path)
    try:
        file_form = InputVideo(title=filename)
        file_form.save()
    except Exception as e:
        return e

    videofile = file_path
    metadata = skvideo.io.ffprobe(videofile)
    str_fps = metadata["video"]['@avg_frame_rate'].split('/')
    fps = float(float(str_fps[0]) / float(str_fps[1]))

    timestamps = [(float(1) / fps)]
    total_frame = float(metadata["video"]["@nb_frames"])
    total_duration = float(metadata["video"]["@duration"])

    sim_cal = int(math.ceil(fps / 10))
    gap = (total_duration / total_frame) * sim_cal * 3 * 1000

    # print(' fps : ', fps, ' | tf : ', total_frame, ' | dur: ', total_duration, ' | frame_hop :', sim_cal, ' |  frame gap in ms : ', gap)
    count = 0
    cele = {}
    ids = []
    cache_embeddings = {}

    videogen = skvideo.io.vreader(videofile)
    for curr_frame in (videogen):
        count = count + 1
        if count % sim_cal == 0:
            timestamps = (float(count) / fps) * 1000  # multiplying to get the timestamps in milliseconds
            try:
                all_faces, all_bb = get_face(img=curr_frame, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                if all_faces is not None:
                    cele_id = []
                    for face, bb in zip(all_faces, all_bb):
                        embedding = embed_image(img=face, session=facenet_persistent_session, images_placeholder=images_placeholder, embeddings=embeddings,
                                                phase_train_placeholder=phase_train_placeholder, image_size=image_size)
                        id_name = ''
                        if embedding_dict:
                            if cache_embeddings:
                                id_name = identify_face(embedding=embedding, embedding_dict=cache_embeddings)
                                if id_name == "Unknown":
                                    id_name = identify_face(embedding=embedding, embedding_dict=embedding_dict)
                                    if id_name != "Unknown":
                                        cache_embeddings[id_name] = embedding
                            else:
                                id_name = identify_face(embedding=embedding, embedding_dict=embedding_dict)
                                cache_embeddings[id_name] = embedding

                            if(str(id_name) not in ids):
                                ids.append(str(id_name))
                                cele[str(id_name)] = []
                            cele_id.append(id_name)
                            cele[str(id_name)].append(timestamps)
                else:
                    return 'error no faces '
            except Exception as e:
                return e

    output_dur = time_dura(cele, gap)
    try:
        with open(os.path.join(MEDIA_ROOT, 'output/video', filename.split('.')[0] + '.json'), 'w') as fp:
            json.dump(output_dur, fp)
    except Exception as e:
        print(e)
        pass
    file_form.isProcessed = True
    file_form.save()
    return output_dur


def createEmbedding(request, filename):
    file = request.FILES['file']
    if file and allowed_file(filename=filename, allowed_set=allowed_set):
        filename = secure_filename(filename=filename).replace('_', ' ').split('.')[0].title()
        unid = uuid.uuid4().hex
        try:
            filepath = "/media/face/" + str(unid) + '.jpg'
            file_form = InputEmbed(id=unid, title=filename, fileurl=filepath)
            file_form.save()
        except Exception as e:
            return (e)

        img = imread(fname=file, mode='RGB')
        if (img.shape[2] == 4):
            img = img[..., :3]

        try:
            face, bb = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
            if face is not None:
                embedding = embed_image(img=face[0], session=facenet_persistent_session, images_placeholder=images_placeholder, embeddings=embeddings,
                                        phase_train_placeholder=phase_train_placeholder, image_size=image_size)
                save_face(img=face[0], filename=unid)
                save_embedding(embedding=embedding, filename=filename, embeddings_path=embeddings_path)

                return 'success'
            else:
                return {"Error": 'No face found'}
        except Exception as e:
            return e
    else:
        return {"Error": 'bad file format'}


def stream_video_download(url, filename):
    output_dir = "{}/{}/".format(MEDIA_ROOT, 'videos')
    command = "youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'  \"{}\" -o {}.mp4".format(url, filename)
    try:
        download = subprocess.Popen(shlex.split(command), cwd=output_dir)
        download.wait()
    except Exception as e:
        return e
