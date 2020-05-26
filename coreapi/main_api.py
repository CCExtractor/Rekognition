import os
import math
import uuid
import json
import skvideo.io
import subprocess
import shlex
import cv2
from skimage.io import imread
import urllib.parse
from werkzeug.utils import secure_filename
from Rekognition.settings import MEDIA_ROOT
from corelib.facenet.utils import (get_face, embed_image, save_embedding,
                                   identify_face, allowed_file, time_dura,
                                   handle_uploaded_file, save_face,
                                   img_standardize)
from corelib.constant import (pnet, rnet, onet, facenet_persistent_session,
                              phase_train_placeholder, embeddings,
                              images_placeholder, image_size, allowed_set,
                              embeddings_path, embedding_dict,
                              Facial_expression_class_names, nsfw_class_names,
                              base_url, face_exp_url, nsfw_url)
from .models import InputImage, InputVideo, InputEmbed, SimilarFaceInImage
import numpy as np
import requests
from skimage.transform import resize
from coreapi.serializers import ImageFrNetworkChoices
from corelib.RetinaFace.retina_net import FaceDetectionRetina


def faceexp(cropped_face):
    """     Facical Expression Recognition of faces in image
    Args:
            *   cropped_face: numpy array of cropped face
    Workflow:
            *   A numpy array of a cropped face is taken as input (RGB).
                inference input dimension requires dimension of (1,100,100,3)
                therefore the RGB input is first converted to grayscale image
                followed by normalizing & resizing to required input dimension.
            *   Now the processed output is further processed to make it a
                json format which is compatible to TensorFlow Serving input.
            *   Then a http post request is made at localhost:8501.
                The post request contain data and headers.
            *   Incase of any exception, it return empty string.
            *   output from TensorFlow Serving is then parsed and a
                dictionary is defined which keeps the facial expression name
                as key and prediction's output as value. The prediction values
                are floating point values which tells the probability of the
                particular facial expression.
    Returns:
            *   Dictionary having all the faces and corresponding facial
                expression and it's values.
    """
    img = cv2.resize(cropped_face, (100, 100), interpolation=cv2.INTER_AREA)
    img = np.array(img).reshape((1, 100, 100, 3))
    img = img / 255
    data = json.dumps({"signature_name": "serving_default",
                       "instances": img.tolist()})
    try:
        headers = {"content-type": "application/json"}
        url = urllib.parse.urljoin(base_url, face_exp_url)
        json_response = requests.post(url, data=data, headers=headers)
        print(json.loads(json_response.text))
    except Exception as e:
        print(e, "\n TensorFlow Serving is not working properly")
        return " "
    predictions = json.loads(json_response.text)["predictions"]
    final_result = {}
    for key, value in zip(Facial_expression_class_names, predictions[0]):
        final_result[key] = value
    return final_result


def nsfwclassifier(request, filename):
    """     NSFW classifier of images
    Args:
            *   request: Post https request containing a image file
            *   filename: filename of the video
    Workflow:
            *   A numpy array of an image is taken as input (RGB).
                inference input dimension requires dimension of (64,64,1)
                and therefore the RGB input is resized to the required
                input dimension.
            *   Now the processed output is further processed to make it a
                json format which is compatible to TensorFlow Serving input.
            *   Then a http post request is made at localhost:8501 .
                The post request contain data and headers.
            *   Incase of any exception, it return empty string.
            *   output from TensorFlow Serving is then parsed and a dictionary
                is defined which keeps classes and probabilities as the two
                keys. Value of Classes is the class with maximum probability
                and value of probabilities is a dictionary of classes with
                their respective probabilities
    Returns:
            *   Dictionary having the class with maximum probability and
                probabilities of all classes.
    """

    file_path = os.path.join(MEDIA_ROOT, 'images/' + filename)
    handle_uploaded_file(request.FILES['file'], file_path)

    img = imread(file_path)
    img = resize(img, (64, 64), anti_aliasing=True, mode='constant')
    if (img.shape[2] == 4):
        img = img[..., :3]

    data = np.asarray(img, dtype="float32")
    data = img_standardize(data)
    image_data = data.astype(np.float16, copy=False)
    url = urllib.parse.urljoin(base_url, nsfw_url)
    jsondata = json.dumps({"inputs": [image_data.tolist()]})
    try:
        response = requests.post(url, data=jsondata)
    except Exception as e:
        print(e, "\n TensorFlow Serving is not working properly")
        return " "
    data = response.json()
    outputs = data['outputs']
    predict_result = {"classes": nsfw_class_names.get(outputs['classes'][0])}
    predict_result['probabilities'] = {nsfw_class_names.get(i): l for i,
                                       l in enumerate(outputs['probabilities'][0])}
    return predict_result


def facerecogniseinimage(request, filename, network):
    """     Face Recognition in image
    Args:
            *   request: Post https request containing a image file
            *   filename: filename of the video
            *   network: Network architecture to be used for face detection
    Workflow:
            *   Image file is first saved into images which is subfolder of
                MEDIA_ROOT directory.
            *   If there is any file in the post request and file extension is
                mentioned in allowed_set then it is allowed for further
                processing else the returns an error.
            *   Then, the image is converted into numpy array, followed by
                reshaping the image dimension if it contains 4 channels.
                It is actually done to process .png files.
            *   Then, all the faces present in an image is extracted along
                with corresponding boundingbox using method get_face.
            *   if number of faces is greater than 0 then for each face and
                corresponding bounding box is taken for further processing.
            *   embedding for each face is created with help of embed_image
                and returned to the vairable embedding
            *   Now this embedding is compared with already available
                embeddings, It returns the name of the embedding which has the
                lowest difference in them else if it crosses the limit then
                'unknown' id returned.
            *   Next step is to get the facial expression of the face using
                the method faceexp.
            *   Then, all the information which currently includes face id,
                bounding box and facial expression is saved into a dictionary.
            *   Information about the face is saved into database.
    Returns:
            *   Dictionary having all the faces and corresponding bounding
                boxes with facial expression
    """
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

            if network == ImageFrNetworkChoices[0]:
                all_faces, all_bb = FaceDetectionRetina().get_face(file_path)

            else:
                all_faces, all_bb = get_face(img=img, pnet=pnet,
                                             rnet=rnet, onet=onet,
                                             image_size=image_size)

            all_face_arr = []

            if all_faces is not None:
                for img, bb in zip(all_faces, all_bb):
                    embedding = embed_image(img=img,
                                            session=facenet_persistent_session,
                                            images_placeholder=images_placeholder,
                                            embeddings=embeddings,
                                            phase_train_placeholder=phase_train_placeholder,
                                            image_size=image_size)

                    if embedding_dict:
                        id_name = identify_face(embedding=embedding,
                                                embedding_dict=embedding_dict)
                        facial_expression = faceexp(img)

                        bounding_box = {"top": bb[1], "bottom": bb[3],
                                        "left": bb[0], "right": bb[2]}
                        face_dict = {"Identity": id_name,
                                     "Bounding Boxes": bounding_box,
                                     "Facial Expression": facial_expression, }
                        all_face_arr.append(face_dict)
                file_form.is_processed = True
                file_form.save()
                return {"Faces": all_face_arr, }

            else:
                return 'error no faces'
        except Exception as e:
            raise e
            return 'error occured'
    else:
        return {"Error": 'bad file format'}


def facerecogniseinvideo(request, filename):
    """     Face Recognition in Video
    Args:
            *   request: Post https request containing a video file
            *   filename: filename of the video
    Workflow
            *   Video file is first saved into videos which is subfolder of
                MEDIA_ROOT directory.
            *   Information about the video is saved into database
            *   Using skvideo meta information of the video is extracted
            *   With the help of extracted metadata frame/sec (fps) is
                calculated and with this frame_hop is calculated.
                Now this frame_hop is actually useful in decreasing the
                number of frames to be processed, say for example if a video
                is of 30 fps then with frame_hop of 2, every third frame is
                processed, It reduces the computation work. Ofcourse for more
                accuracy the frame_hop can be reduced, but It is observed that
                this affect the output very little.
            *   Now videofile is read using skvideo.io.vreader(), Now, each
                frame is read from videogen. Now timestamp of the particular
                image or face calculated using above metadata.
            *   Now a locallist is maintained which keeps the all face ids.
            *   Now Faces and corresponding boundingbox is being calculated,
                if there is even a single face in output then it is taken for
                further processing like creating embedding for each face.
            *   After embedding is created for any particular face then ,
                It is checked whether the global embedding_dict contains
                any already embedding or not with which the current embedding
                is to be compared. Initially cache_embeddings is empty
                therefore else condition is true and executed first. cache
                embedding is created to reduce computation by a huge margin.
                It checks whether the embeddings of the faces in current
                frames were present in the previous frames or not. If there is
                a hit then it moves to identify the next face else It looks
                globally for the face embeddings and on hit, current face
                embedding is added to the local cache_embeddings. This works
                since it embedding is first checked in local dictionary and
                no need to compare with all the available embeddings in the
                global embedding_dict. In videos it is very likely that the
                face might be distorted in next frame and this may bring more
                error . By this method it also minimizes this error by a great
                margin.
            *   After that a dictionary is maintained which keeps the face id
                as key and corrensponding timestamps of the faces in array
            *   time_dura coalesces small timestamps into a time interval
                duration of each face ids.
    Returns:
            *   Dictionary having all the faces & corresponding time durations
    """
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

    frame_hop = int(math.ceil(fps / 10))
    gap_in_sec = (total_duration / total_frame) * frame_hop * 3 * 1000

    count = 0
    cele = {}
    ids = []
    cache_embeddings = {}

    videogen = skvideo.io.vreader(videofile)
    for curr_frame in (videogen):
        count = count + 1
        if count % frame_hop == 0:
            # multiplying to get the timestamps in milliseconds
            timestamps = (float(count) / fps) * 1000
            try:
                all_faces, all_bb = get_face(img=curr_frame,
                                             pnet=pnet, rnet=rnet,
                                             onet=onet, image_size=image_size)
                if all_faces is not None:
                    cele_id = []
                    for face, bb in zip(all_faces, all_bb):
                        embedding = embed_image(img=face,
                                                session=facenet_persistent_session,
                                                images_placeholder=images_placeholder,
                                                embeddings=embeddings,
                                                phase_train_placeholder=phase_train_placeholder,
                                                image_size=image_size)
                        id_name = ''
                        if embedding_dict:
                            if cache_embeddings:
                                id_name = identify_face(embedding=embedding,
                                                        embedding_dict=cache_embeddings)
                                if id_name == "Unknown":
                                    id_name = identify_face(embedding=embedding,
                                                            embedding_dict=embedding_dict)
                                    if id_name != "Unknown":
                                        cache_embeddings[id_name] = embedding
                            else:
                                id_name = identify_face(embedding=embedding,
                                                        embedding_dict=embedding_dict)
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

    output_dur = time_dura(cele, gap_in_sec)
    try:
        with open(os.path.join(MEDIA_ROOT, 'output/video', filename.split('.')[0] + '.json'), 'w') as fp:
            json.dump(output_dur, fp)
    except Exception as e:
        print(e)
        pass
    file_form.is_processed = True
    file_form.save()
    return output_dur


def createembedding(request, filename):
    """      To create face embedding
    Args:
            *   request: Post https request containing a image file
            *   filename: filename of the video
    Workflow
            *   First it checks whether is there any image file in the post
                request.
            *   Image information is saved in the database
            *   face and corresponding bounding box is extracted followed by
                creating and saving bounding box.
    Returns:
            *   success flag
    """
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
            face, bb = get_face(img=img, pnet=pnet,
                                rnet=rnet, onet=onet,
                                image_size=image_size)
            if face is not None:
                embedding = embed_image(img=face[0],
                                        session=facenet_persistent_session,
                                        images_placeholder=images_placeholder,
                                        embeddings=embeddings,
                                        phase_train_placeholder=phase_train_placeholder,
                                        image_size=image_size)
                save_face(img=face[0], where='face', filename=unid)
                save_embedding(embedding=embedding, filename=filename,
                               embeddings_path=embeddings_path)

                return 'success'
            else:
                return {"Error": 'No face found'}
        except Exception as e:
            return e
    else:
        return {"Error": 'bad file format'}


def stream_video_download(url, filename):
    """     To download a youtube video
    Args:
            *   url: url of the youtube video
            *   filename: filename of the video
    Workflow
            *   It uses youtube-dl to download the best available
    Returns:
            *   Nothing
    """
    output_dir = "{}/{}/".format(MEDIA_ROOT, 'videos')
    command = "youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'  \"{}\" -o {}.mp4".format(url, filename)
    try:
        download = subprocess.Popen(shlex.split(command), cwd=output_dir)
        download.wait()
    except Exception as e:
        return e


def process_streaming_video(url, filename):
    output_dir = "{}/{}/".format(MEDIA_ROOT, 'videos')
    try:
        stream_video_download(url, filename)
    except Exception as e:
        return e

    file_dir = os.path.join(output_dir, filename + '.mp4')
    files = {'file': open(file_dir, 'rb')}
    result = requests.post('http://localhost:8000/api/old_video/',
                           files=files)
    return result


def similarface(request, filename):
    """     Face Recognition in image
    Args:
            *   request: Post https request containing a image file
            *   filename: filename of the image
    Workflow:
            *   Image folder is created inside similarFace which is subfolder
                of MEDIA_ROOT directory.
            *	Reference file and Compare file are saved similarFace folder
            *   Bounding boxes of faces are found in reference image using
                get_face
            *	If no faces are present in reference image return error
            *   Else, save face with highest confidence in the current folder
            *   Then, generate embedding of the face found above using
                embed_image
            *   Bounding boxes of faces are found in compare image using
                get_face
            *	If no faces are present in reference image return error
            *	Else, save all faces in the current folder
            *	Then, generate embedding of every image using embed_image
                and compare embedding of each image with reference image
                using identify_face
            *	If match is found return the request string along with the id
                of the image
            *	else, return the request string along with "None"
    Returns:
            *   Dictionary having all the faces and corresponding bounding
                boxes with facial expression
    """
    file_folder = MEDIA_ROOT + '/' + 'similarFace' + '/' + str(filename.split('.')[0])

    if not os.path.exists(file_folder):
        os.makedirs(file_folder)

    handle_uploaded_file(request.FILES['file'], file_folder + '/referenceImage.jpg')
    handle_uploaded_file(request.FILES['compareImage'], file_folder + '/compareImage.jpg')

    try:
        # filepath = "/media/similarFace/" + str(filename.split('.')[0]) +'/'+'referenceImage.jpg'
        file_form = SimilarFaceInImage(title=filename.split('.')[0])
        file_form.save()
    except Exception as e:
        return (e)

    ref_img = request.FILES['file']
    com_img = request.FILES['compareImage']

    ref_img = imread(fname=ref_img, mode='RGB')
    if (ref_img.shape[2] == 4):
        ref_img = ref_img[..., :3]

    com_img = imread(fname=com_img, mode='RGB')
    if (com_img.shape[2] == 4):
        com_img = com_img[..., :3]

    ref_img_face, ref_img_bb = get_face(img=ref_img, pnet=pnet,
                                        rnet=rnet, onet=onet,
                                        image_size=image_size)
    if not ref_img_face:
        return([str(filename.split('.')[0]), "No Face in reference image"])
    save_face(ref_img_face[0], file_folder, filename.split('.')[0])
    ref_face_embedding = embed_image(img=ref_img_face[0],
                                     session=facenet_persistent_session,
                                     images_placeholder=images_placeholder,
                                     embeddings=embeddings,
                                     phase_train_placeholder=phase_train_placeholder,
                                     image_size=image_size)

    try:
        all_faces, all_bb = get_face(img=com_img, pnet=pnet,
                                     rnet=rnet, onet=onet,
                                     image_size=image_size)
        all_face_dict = {}
        if all_faces is not None:
            face_no = 0
            for img, bb in zip(all_faces, all_bb):
                save_face(img=img, where=file_folder, filename=face_no)

                embedding = embed_image(img=img,
                                        session=facenet_persistent_session,
                                        images_placeholder=images_placeholder,
                                        embeddings=embeddings,
                                        phase_train_placeholder=phase_train_placeholder,
                                        image_size=image_size)
                all_face_dict[face_no] = embedding
                face_no += 1
        id_name = identify_face(embedding=ref_face_embedding,
                                embedding_dict=all_face_dict)

        if id_name != "Unknown":
            file_form.similarwith = id_name
            file_form.save()
            return([str(filename.split('.')[0]), id_name])
        else:
            file_form.similarwith = "None"
            file_form.save()
            return([str(filename.split('.')[0]), "None"])

    except Exception as e:
        return (e)
