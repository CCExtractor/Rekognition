from celery import shared_task
from Rekognition.celery import app
import os
import math
import json

import skvideo.io
from Rekognition.settings import MEDIA_ROOT
from corelib.facenet.utils import (get_face, embed_image, load_embeddings,
                                   identify_face, time_dura, handle_uploaded_file)
from corelib.constant import (pnet, rnet, onet, facenet_persistent_session, phase_train_placeholder,
                              embeddings, images_placeholder, image_size, embeddings_path)
from .models import InputVideo


@shared_task
def addi(x, y):
    print('*' * (x + y))
    return x + y


@shared_task
def CFRVideo(file_path, filename):
    # file_path = os.path.join(MEDIA_ROOT, 'videos/' + filename)
    # handle_uploaded_file(request.FILES['file'], file_path)
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

    print(' fps : ', fps, ' | tf : ', total_frame, ' | dur: ', total_duration, ' | frame_hop :', sim_cal, ' |  frame gap in ms : ', gap)
    count = 0
    cele = {}
    ids = []
    embedding_dict = load_embeddings(embeddings_path)
    cache_embeddings = {}

    videogen = skvideo.io.vreader(videofile)
    for curr_frame in (videogen):
        count = count + 1
        if count % sim_cal == 0:
            print("hola")
            timestamps = (float(count) / fps) * 1000  # multiplying to get the timestamps in milliseconds
            try:
                print("ppppppppp", len(embedding_dict))
                all_faces, all_bb = get_face(img=curr_frame, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)
                print("66666", len(all_faces))
                if all_faces is not None:
                    print("la")
                    cele_id = []
                    for face, bb in zip(all_faces, all_bb):
                        embedding = embed_image(img=face, session=facenet_persistent_session, images_placeholder=images_placeholder, embeddings=embeddings,
                                                phase_train_placeholder=phase_train_placeholder, image_size=image_size)
                        id_name = ''
                        print("test")
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
                        print(cele_id)
                    print("bhola")
                else:
                    return 'error no faces '
            except Exception as e:
                print(e)
                return e

    output_dur = time_dura(cele, gap)
    with open(os.path.join(MEDIA_ROOT, 'result.json'), 'w') as fp:
        json.dump(output_dur, fp)
    return output_dur
