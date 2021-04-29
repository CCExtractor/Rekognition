import os
import math
import uuid
import json
import skvideo.io
import subprocess
import shlex
import cv2
import wordninja
from skimage.io import imread
import urllib.parse
from werkzeug.utils import secure_filename
from Rekognition.settings import MEDIA_ROOT
from corelib.CRNN import CRNN_utils
from corelib.facenet.utils import (get_face, embed_image, save_embedding,
                                   identify_face, allowed_file, time_dura,
                                   handle_uploaded_file, save_face,
                                   img_standardize)
from corelib.EAST.EAST_utils import (preprocess, sort_poly,
                                     postprocess)
from corelib.constant import (pnet, rnet, onet, facenet_persistent_session,
                              phase_train_placeholder, embeddings,
                              images_placeholder, image_size, allowed_set,
                              embeddings_path, embedding_dict,
                              Facial_expression_class_names, nsfw_class_names,
                              base_url, face_exp_url, nsfw_url, text_reco_url,
                              char_dict_path, ord_map_dict_path, text_detect_url,
                              coco_names_path, object_detect_url, scene_detect_url,
                              scene_labels_path)
from corelib.utils import ImageFrNetworkChoices, get_class_names, bb_to_cv, get_classes
from coreapi.models import InputImage, InputVideo, InputEmbed, SimilarFaceInImage
from logger.logging import RekogntionLogger
import numpy as np
import requests
from skimage.transform import resize
from corelib.RetinaFace.retina_net import FaceDetectionRetina
from django.db import IntegrityError, DatabaseError


logger = RekogntionLogger(name="main_api")


def text_reco(image):
    """     Scene Text Recognition
    Args:
            *   image: numpy array of cropped text
    Workflow:
            *   A numpy array of a cropped text is taken as input
                inference input dimension requires dimension of (100,32)
                therefore the input is first resizing to required
                input dimension and then normalized.
            *   Now the processed output is further processed to make it a
                json format which is compatible to TensorFlow Serving input.
            *   Then a http post request is made at localhost:8501.
                The post request contain data and headers.
            *   Incase of any exception, it return relevant error message.
            *   output from TensorFlow Serving is further processed using
                wordninja and then returned as a dictionary which keeps Text
                as key and processed output as value.
    Returns:
            *   Dictionary having text as Key and processed output as value.
    """

    logger.info(msg="text_reco called")
    image = cv2.resize(image, tuple((100, 32)), interpolation=cv2.INTER_LINEAR)
    image = np.array(image, np.float32) / 127.5 - 1.0
    data = json.dumps({"inputs": [image.tolist()]})
    try:
        headers = {"content-type": "application/json"}
        url = urllib.parse.urljoin(base_url, text_reco_url)
        json_response = requests.post(url, data=data, headers=headers)
    except requests.exceptions.HTTPError as errh:
        logger.error(msg=errh)
        return {"Error": "An HTTP error occurred."}
    except requests.exceptions.ConnectionError as errc:
        logger.error(msg=errc)
        return {"Error": "A Connection error occurred."}
    except requests.exceptions.Timeout as errt:
        logger.error(msg=errt)
        return {"Error": "The request timed out."}
    except requests.exceptions.TooManyRedirects as errm:
        logger.error(msg=errm)
        return {"Error": "Bad URL"}
    except requests.exceptions.RequestException as err:
        logger.error(msg=err)
        return {"Error": "Facial Expression Recognition Not Working"}
    except Exception as e:
        logger.error(msg=e)
        return {"Error": "Facial Expression Recognition Not Working"}
    predictions = json.loads(json_response.text).get("outputs", "Bad request made.")
    codec = CRNN_utils._FeatureIO(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path,
    )

    preds = codec.sparse_tensor_to_str_for_tf_serving(
        decode_indices=predictions['decodes_indices'],
        decode_values=predictions['decodes_values'],
        decode_dense_shape=predictions['decodes_dense_shape'],
    )[0]
    preds = ' '.join(wordninja.split(preds))
    return {"Text": preds}


def text_detect(input_file, filename):
    """     Scene Text Detection
    Args:
            *   input_file: Contents of the input image file
            *   filename: filename of the image
    Workflow:
            *   A numpy array of an image with text is taken as input
                inference input dimension requires dimension to be in a
                multiple of 32 therefore the input is first resized to
                 required input dimension and then normalized.
            *   Now the processed output is further processed to make it a
                json format which is compatible to TensorFlow Serving input.
            *   Then a http post request is made at localhost:8501.
                The post request contain data and headers.
            *   Incase of any exception, it return relevant error message.
            *   output from TensorFlow Serving is further processed using
                Locality-Aware Non-Maximum Suppression (LANMS)
            *   Calls to text_reco are made for each of these detected
                boxes whic returns the recognized text in these boxes
            *   A list is maintained with each element being a dictionary
                with Boxes as one of the keys and coordinates of the
                detected bounding box as the value and Text as another key
                with the text recognized by text_reco as value
            *   A dictionay is returned with Texts as key and the list
                generated above as value
    Returns:
            *   Dictionary having Texts as the key and list of dictionaries
                as the value where the dictinary elemet has Boxes and Text
                as keys and coordinates of bounding boxes and recognized
                text of that box as the respective value
    """

    logger.info(msg="text_detect called")
    file_path = os.path.join(MEDIA_ROOT, 'text', filename)
    handle_uploaded_file(input_file, file_path)
    img = cv2.imread(file_path)[:, :, ::-1]
    img_resized, (ratio_h, ratio_w) = preprocess(img)
    img_resized = (img_resized / 127.5) - 1
    data = json.dumps({"signature_name": "serving_default",
                       "inputs": [img_resized.tolist()]})
    try:
        headers = {"content-type": "application/json"}
        url = urllib.parse.urljoin(base_url, text_detect_url)

        json_response = requests.post(url, data=data, headers=headers)
    except requests.exceptions.HTTPError as errh:
        logger.error(msg=errh)
        return {"Error": "An HTTP error occurred."}
    except requests.exceptions.ConnectionError as errc:
        logger.error(msg=errc)
        return {"Error": "A Connection error occurred."}
    except requests.exceptions.Timeout as errt:
        logger.error(msg=errt)
        return {"Error": "The request timed out."}
    except requests.exceptions.TooManyRedirects as errm:
        logger.error(msg=errm)
        return {"Error": "Bad URL"}
    except requests.exceptions.RequestException as err:
        logger.error(msg=err)
        return {"Error": "Facial Expression Recognition Not Working"}
    except Exception as e:
        logger.error(msg=e)
        return {"Error": e}
    predictions = json.loads(json_response.text)["outputs"]
    score_map = np.array(predictions["pred_score_map/Sigmoid:0"], dtype="float64")
    geo_map = np.array(predictions["pred_geo_map/concat:0"], dtype="float64")

    boxes = postprocess(score_map=score_map, geo_map=geo_map)
    result_boxes = []
    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h
        for box in boxes:
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            result_boxes.append(box)
    result = []
    for box in result_boxes:
        top_left_x, top_left_y, bot_right_x, bot_right_y = bb_to_cv(box)
        text = text_reco(img[top_left_y - 2:bot_right_y + 2, top_left_x - 2:bot_right_x + 2]).get("Text")
        result.append({"Boxes": box, "Text": text})
    return {"Texts": result}


def text_detect_video(input_file, filename):
    """     Scene Text Detection in video
    Args:
            *   input_file: Contents of the input video file
            *   filename: filename of the video
    Workflow:
            *   uploaded file is read using opencv and gets processed
                frame by frame
            *   inference input dimension requires dimension to be in a
                multiple of 32 therefore each frame is first resized to
                required input dimension and then normalized.
            *   Now the processed output is further processed to make it a
                json format which is compatible to TensorFlow Serving input.
            *   Then a http post request is made at localhost:8501.
                The post request contain data and headers.
            *   Incase of any exception, it return relevant error message.
            *   output from TensorFlow Serving is further processed using
                Locality-Aware Non-Maximum Suppression (LANMS)
            *   Calls to text_reco are made for each of these detected
                boxes whic returns the recognized text in these boxes
            *   A list is maintained with each element being a dictionary
                with Boxes as one of the keys and coordinates of the
                detected bounding box as the value and Text as another key
                with the text recognized by text_reco as value
            *   Result of every frame is stored in another list
            *   A dictionay is returned with Texts as key and the list
                generated above as value
    Returns:
            *   Dictionary having Texts as the key and list of dictionaries
                as the value where the dictinary elemet has Boxes and Text
                as keys and coordinates of bounding boxes and recognized
                text of that box as the respective value for every frame
    """

    logger.info(msg="text_detect_video called")
    file_path = os.path.join(MEDIA_ROOT, 'text', filename)
    handle_uploaded_file(input_file, file_path)
    video_result = []
    vid = cv2.VideoCapture(file_path)
    while(vid.isOpened()):
        ret, img = vid.read()
        if ret:
            img = img[:, :, ::-1]
            img_resized, (ratio_h, ratio_w) = preprocess(img)
            img_resized = (img_resized / 127.5) - 1
            data = json.dumps({"signature_name": "serving_default",
                               "inputs": [img_resized.tolist()]})
            try:
                headers = {"content-type": "application/json"}
                url = urllib.parse.urljoin(base_url, text_detect_url)

                json_response = requests.post(url, data=data, headers=headers)
            except requests.exceptions.HTTPError as errh:
                logger.error(msg=errh)
                return {"Error": "An HTTP error occurred."}
            except requests.exceptions.ConnectionError as errc:
                logger.error(msg=errc)
                return {"Error": "A Connection error occurred."}
            except requests.exceptions.Timeout as errt:
                logger.error(msg=errt)
                return {"Error": "The request timed out."}
            except requests.exceptions.TooManyRedirects as errm:
                logger.error(msg=errm)
                return {"Error": "Bad URL"}
            except requests.exceptions.RequestException as err:
                logger.error(msg=err)
                return {"Error": "Facial Expression Recognition Not Working"}
            except Exception as e:
                logger.error(msg=e)
                return {"Error": e}
            predictions = json.loads(json_response.text)["outputs"]
            score_map = np.array(predictions["pred_score_map/Sigmoid:0"], dtype="float64")
            geo_map = np.array(predictions["pred_geo_map/concat:0"], dtype="float64")

            boxes = postprocess(score_map=score_map, geo_map=geo_map)
            result_boxes = []
            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h
                for box in boxes:
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    result_boxes.append(box)
            result = []
            for box in result_boxes:
                top_left_x, top_left_y, bot_right_x, bot_right_y = bb_to_cv(box)
                text = text_reco(img[top_left_y - 2:bot_right_y + 2, top_left_x - 2:bot_right_x + 2]).get("Text")
                result.append({"Boxes": box, "Text": text})
            video_result.append(result)
        else:
            break
    return {"Texts": video_result}


def scene_detect(input_file, filename):
    """     Scene Text Detection
    Args:
            *   input_file: Contents of the input image file
            *   filename: filename of the image
    Workflow:
            *   A numpy array of an image is taken as input (RGB).
            *   inference input dimension requires dimension of (224,224)
                therefore the input is first resizing to required dimension
            *   Now the processed output is further processed to make it a
                json format which is compatible to TensorFlow Serving input.
            *   Then a http post request is made at localhost:8501.
                The post request contain data and headers.
            *   Incase of any exception, it return relevant error message.
            *   output from TensorFlow Serving is further processed to find
                the classes that have been recognized
            *   A list is maintained with each element being a dictionary
                with Scene as one of the keys and the detected class as the
                value and Score as another key with the probability of that
                class as value
            *   A dictionay is returned with Scenes as key and the list
                generated above as value
    Returns:
            *   Dictionary having Scenes as the key and list of dictionaries
                as the value where the dictinary elemet has Scene and Score
                as keys and detected class and confidence score of that
                class as the respective value
    """

    logger.info(msg="scene_detect called")
    file_path = os.path.join(MEDIA_ROOT, 'scene', filename)
    handle_uploaded_file(input_file, file_path)
    img = cv2.imread(file_path)[:, :, ::-1]
    img_resized = cv2.resize(img, (224, 224))
    data = json.dumps({"signature_name": "serving_default",
                       "inputs": [img_resized.tolist()]})
    try:
        headers = {"content-type": "application/json"}
        url = urllib.parse.urljoin(base_url, scene_detect_url)

        json_response = requests.post(url, data=data, headers=headers)
    except requests.exceptions.HTTPError as errh:
        logger.error(msg=errh)
        return {"Error": "An HTTP error occurred."}
    except requests.exceptions.ConnectionError as errc:
        logger.error(msg=errc)
        return {"Error": "A Connection error occurred."}
    except requests.exceptions.Timeout as errt:
        logger.error(msg=errt)
        return {"Error": "The request timed out."}
    except requests.exceptions.TooManyRedirects as errm:
        logger.error(msg=errm)
        return {"Error": "Bad URL"}
    except requests.exceptions.RequestException as err:
        logger.error(msg=err)
        return {"Error": "Facial Expression Recognition Not Working"}
    except Exception as e:
        logger.error(msg=e)
        return {"Error": e}
    predictions = json.loads(json_response.text)["outputs"][0]

    top_preds = np.argsort(predictions)[::-1][0:5]
    top_preds_score = np.sort(predictions)[::-1][0:5]
    classes = get_classes(scene_labels_path)
    result = []
    for i in range(0, 5):
        result.append({"Scene": classes[top_preds[i]], "Score": top_preds_score[i]})
    return {"Scenes": result}


def scene_video(input_file, filename):
    """     Scene classification in videos
    Args:
            *   input_file: Contents of the input video file
            *   filename: filename of the video
    Workflow:
            *   uploaded file is read using opencv and gets processed
                frame by frame
            *   A numpy array of an image is taken as input (RGB).
            *   inference input dimension requires dimension of (224,224)
                therefore the input is first resizing to required dimension
            *   Now the processed output is further processed to make it a
                json format which is compatible to TensorFlow Serving input.
            *   Then a http post request is made at localhost:8501.
                The post request contain data and headers.
            *   Incase of any exception, it return relevant error message.
            *   output from TensorFlow Serving is further processed to find
                the classes that have been recognized
            *   A list is maintained with each element being a dictionary
                with Scene as one of the keys and the detected class as the
                value and Score as another key with the probability of that
                class as value
            *   Result of every frame is stored in a list
            *   A dictionary is returned with Result as key and the list
                generated above as the value
    Returns:
            *   Dictionary having Result as Key and list of dictionaries
                as the value where the dictionary element has Scenes as
                the key and list of dictionaries as the value and these
                dictinary elemet has Scene and Score as keys and detected
                class and confidence score of that class as the respective value
    """

    logger.info(msg="scene_video called")
    file_path = os.path.join(MEDIA_ROOT, 'scene', filename)
    handle_uploaded_file(input_file, file_path)
    vid = cv2.VideoCapture(file_path)
    video_result = []
    while(vid.isOpened()):
        ret, image = vid.read()
        if ret:
            img = image[:, :, ::-1]
            img_resized = cv2.resize(img, (224, 224))
            data = json.dumps({"signature_name": "serving_default",
                               "inputs": [img_resized.tolist()]})
            try:
                headers = {"content-type": "application/json"}
                url = urllib.parse.urljoin(base_url, scene_detect_url)

                json_response = requests.post(url, data=data, headers=headers)
            except requests.exceptions.HTTPError as errh:
                logger.error(msg=errh)
                return {"Error": "An HTTP error occurred."}
            except requests.exceptions.ConnectionError as errc:
                logger.error(msg=errc)
                return {"Error": "A Connection error occurred."}
            except requests.exceptions.Timeout as errt:
                logger.error(msg=errt)
                return {"Error": "The request timed out."}
            except requests.exceptions.TooManyRedirects as errm:
                logger.error(msg=errm)
                return {"Error": "Bad URL"}
            except requests.exceptions.RequestException as err:
                logger.error(msg=err)
                return {"Error": "Facial Expression Recognition Not Working"}
            except Exception as e:
                logger.error(msg=e)
                return {"Error": e}
            predictions = json.loads(json_response.text)["outputs"][0]

            top_preds = np.argsort(predictions)[::-1][0:5]
            top_preds_score = np.sort(predictions)[::-1][0:5]
            classes = get_classes(scene_labels_path)
            result = []
            for i in range(0, 5):
                result.append({"Scene": classes[top_preds[i]], "Score": top_preds_score[i]})
            video_result.append(result)
        else:
            break
    return {"Result": video_result}


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

    logger.info(msg="faceexp called")
    img = cv2.resize(cropped_face, (100, 100), interpolation=cv2.INTER_AREA)
    img = np.array(img).reshape((1, 100, 100, 3))
    img = img / 255
    data = json.dumps({"signature_name": "serving_default",
                       "instances": img.tolist()})
    try:
        headers = {"content-type": "application/json"}
        url = urllib.parse.urljoin(base_url, face_exp_url)
        json_response = requests.post(url, data=data, headers=headers)
    except requests.exceptions.HTTPError as errh:
        logger.error(msg=errh)
        return {"Error": "An HTTP error occurred."}
    except requests.exceptions.ConnectionError as errc:
        logger.error(msg=errc)
        return {"Error": "A Connection error occurred."}
    except requests.exceptions.Timeout as errt:
        logger.error(msg=errt)
        return {"Error": "The request timed out."}
    except requests.exceptions.TooManyRedirects as errm:
        logger.error(msg=errm)
        return {"Error": "Bad URL"}
    except requests.exceptions.RequestException as err:
        logger.error(msg=err)
        return {"Error": "Facial Expression Recognition Not Working"}
    except Exception as e:
        logger.error(msg=e)
        return {"Error": "Facial Expression Recognition Not Working"}
    predictions = json.loads(json_response.text).get("predictions", "Bad request made.")
    final_result = {}
    for key, value in zip(Facial_expression_class_names, predictions[0]):
        final_result[key] = value
    return final_result


def nsfwclassifier(input_file, filename):
    """     NSFW classifier of images
    Args:
            *   input_file: Contents of the input image file
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

    logger.info(msg="nsfwclassifier called")
    file_path = os.path.join(MEDIA_ROOT, 'images', filename)

    handle_uploaded_file(input_file, file_path)

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
    except requests.exceptions.HTTPError as errh:
        logger.error(msg=errh)
        return {"Error": "An HTTP error occurred."}
    except requests.exceptions.ConnectionError as errc:
        logger.error(msg=errc)
        return {"Error": "A Connection error occurred."}
    except requests.exceptions.Timeout as errt:
        logger.error(msg=errt)
        return {"Error": "The request timed out."}
    except requests.exceptions.TooManyRedirects as errm:
        logger.error(msg=errm)
        return {"Error": "Bad URL"}
    except requests.exceptions.RequestException as err:
        logger.error(msg=err)
        return {"Error": "NSFW Classification Not Working"}
    except Exception as e:
        logger.error(msg=e)
        return {"Error": "NSFW Classification Not Working"}
    data = response.json()
    outputs = data['outputs']
    predict_result = {"classes": nsfw_class_names.get(outputs['classes'][0])}
    predict_result['probabilities'] = {nsfw_class_names.get(i): l for i,
                                       l in enumerate(outputs['probabilities'][0])}
    return predict_result


def nsfw_video(input_file, filename):
    """     NSFW classification in videos
    Args:
            *   input_file: Contents of the input video file
            *   filename: filename of the video
    Workflow:
            *   uploaded file is read using opencv and gets processed
                frame by frame
            *   The numpy array of each frame is taken as input (RGB).
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
            *   Result of every frame is stored in a list
            *   A dictionary is returned with Result as key and the list
                generated above as the value
    Returns:
            *   Dictionary having Result as Key and list of dictionaries
                as the value where the dictionary element keeps classes
                and probabilities as the two keys. Value of Classes is
                the class with maximum probability and value of probabilities
                is a dictionary of classes with their respective probabilities
    """

    logger.info(msg="nsfw_video called")
    file_path = os.path.join(MEDIA_ROOT, 'nsfw', filename)

    handle_uploaded_file(input_file, file_path)
    vid = cv2.VideoCapture(file_path)
    video_result = []
    while(vid.isOpened()):
        ret, img = vid.read()
        if ret:
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
            except requests.exceptions.HTTPError as errh:
                logger.error(msg=errh)
                return {"Error": "An HTTP error occurred."}
            except requests.exceptions.ConnectionError as errc:
                logger.error(msg=errc)
                return {"Error": "A Connection error occurred."}
            except requests.exceptions.Timeout as errt:
                logger.error(msg=errt)
                return {"Error": "The request timed out."}
            except requests.exceptions.TooManyRedirects as errm:
                logger.error(msg=errm)
                return {"Error": "Bad URL"}
            except requests.exceptions.RequestException as err:
                logger.error(msg=err)
                return {"Error": "NSFW Classification Not Working"}
            except Exception as e:
                logger.error(msg=e)
                return {"Error": "NSFW Classification Not Working"}
            data = response.json()
            outputs = data['outputs']
            predict_result = {"classes": nsfw_class_names.get(outputs['classes'][0])}
            predict_result['probabilities'] = {nsfw_class_names.get(i): l for i,
                                               l in enumerate(outputs['probabilities'][0])}
            video_result.append(predict_result)
        else:
            break
    return {"Result": video_result}


def facerecogniseinimage(input_file, filename, network):
    """     Face Recognition in image
    Args:
            *   input_file: Contents of the input image file
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

    logger.info(msg="facerecogniseinimage called")
    file_path = os.path.join(MEDIA_ROOT, 'images', filename)
    handle_uploaded_file(input_file, file_path)

    if input_file and allowed_file(filename=filename, allowed_set=allowed_set):
        try:
            file_form = InputImage(title=filename)
            file_form.save()
        except IntegrityError as eri:
            logger.error(msg=eri)
            return {"Error": "Integrity Error"}
        except DatabaseError as erd:
            logger.error(msg=erd)
            return {"Error": "Database Error"}
        except Exception as e:
            logger.error(msg=e)
            return {"Error": e}

        img = cv2.imread(file_path)
        # img = imread(fname=input_file, pilmode='RGB')
        if (img.shape[2] == 4):
            img = img[..., :3]

        try:

            if network == ImageFrNetworkChoices.RetinaFace:
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
                logger.error(msg="No Faces")
                return {"Error": "No Faces"}
        except Exception as e:
            logger.error(msg=e)
            return {"Error": e}
    else:
        logger.error(msg="bad file format")
        return {"Error": 'bad file format'}


def facerecogniseinvideo(input_file, filename):
    """     Face Recognition in Video
    Args:
            *   input_file: Contents of the input video file
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

    logger.info(msg="facerecogniseinvideo called")
    file_path = os.path.join(MEDIA_ROOT, 'videos', filename)
    handle_uploaded_file(input_file, file_path)
    try:
        file_form = InputVideo(title=filename)
        file_form.save()
    except IntegrityError as eri:
        logger.error(msg=eri)
        return {"Error": "Integrity Error"}
    except DatabaseError as erd:
        logger.error(msg=erd)
        return {"Error": "Database Error"}
    except Exception as e:
        logger.error(msg=e)
        return {"Error": e}

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
                    logger.error(msg="No Faces")
                    return {"Error": 'No Faces'}
            except Exception as e:
                logger.error(msg=e)
                return {"Error": e}

    output_dur = time_dura(cele, gap_in_sec)
    try:
        with open(os.path.join(MEDIA_ROOT, 'output/video', filename.split('.')[0] + '.json'), 'w') as fp:
            json.dump(output_dur, fp)
    except Exception as e:
        logger.error(msg=e)
        return {"Error": e}
    file_form.is_processed = True
    file_form.save()
    return output_dur


def createembedding(input_file, filename):
    """      To create face embedding
    Args:
            *   input_file: Contents of the input image file
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

    logger.info(msg="createembedding called")
    if input_file and allowed_file(filename=filename, allowed_set=allowed_set):
        filename = secure_filename(filename=filename).replace('_', ' ').split('.')[0].title()
        unid = uuid.uuid4().hex
        try:
            filepath = "/media/face/" + str(unid) + '.jpg'
            file_form = InputEmbed(id=unid, title=filename, fileurl=filepath)
            file_form.save()
        except IntegrityError as eri:
            logger.error(msg=eri)
            return {"Error": "Integrity Error"}
        except DatabaseError as erd:
            logger.error(msg=erd)
            return {"Error": "Database Error"}
        except Exception as e:
            logger.error(msg=e)
            return {"Error": e}

        img = imread(fname=input_file, mode='RGB')
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

                return {"Status": "Embedding Created Succesfully"}
            else:
                return {"Error": 'No Faces'}
        except Exception as e:
            logger.error(msg=e)
            return {"Error": e}
    else:
        logger.error(msg="bad file format")
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

    logger.info(msg="stream_video_download called")
    output_dir = "{}/{}/".format(MEDIA_ROOT, 'videos')
    command = "youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'  \"{}\" -o {}.mp4".format(url, filename)
    try:
        download = subprocess.Popen(shlex.split(command), cwd=output_dir)
        download.wait()
    except Exception as e:
        logger.error(msg=e)
        return e


def process_streaming_video(url, filename):

    logger.info(msg="process_streaming_video called")
    output_dir = "{}/{}/".format(MEDIA_ROOT, 'videos')
    try:
        stream_video_download(url, filename)
    except Exception as e:
        logger.error(msg=e)
        return {"Error": e}

    file_dir = os.path.join(output_dir, filename + '.mp4')
    files = {'file': open(file_dir, 'rb')}
    try:
        result = requests.post('http://localhost:8000/api/old_video/',
                               files=files)
    except requests.exceptions.HTTPError as errh:
        logger.error(msg=errh)
        return {"Error": "An HTTP error occurred."}
    except requests.exceptions.ConnectionError as errc:
        logger.error(msg=errc)
        return {"Error": "A Connection error occurred."}
    except requests.exceptions.Timeout as errt:
        logger.error(msg=errt)
        return {"Error": "The request timed out."}
    except requests.exceptions.TooManyRedirects as errm:
        logger.error(msg=errm)
        return {"Error": "Bad URL"}
    except requests.exceptions.RequestException as err:
        logger.error(msg=err)
        return {"Error": "Video Processing Not Working"}
    except Exception as e:
        logger.error(msg=e)
        return {"Error": "Video Processing Not Working"}
    return result


def similarface(reference_img, compare_img, filename):
    """     Face Recognition in image
    Args:
            *   reference_img: Contents of the input reference image file
            *   compare_img: Contents of the input compare image file
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

    logger.info(msg="similarface called")
    file_folder = MEDIA_ROOT + '/' + 'similarFace' + '/' + str(filename.split('.')[0])

    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    handle_uploaded_file(reference_img, os.path.join(file_folder, 'referenceImage.jpg'))
    handle_uploaded_file(compare_img, os.path.join(file_folder, 'compareImage.jpg'))

    try:
        # filepath = "/media/similarFace/" + str(filename.split('.')[0]) +'/'+'referenceImage.jpg'
        file_form = SimilarFaceInImage(title=filename.split('.')[0])
        file_form.save()
    except IntegrityError as eri:
        logger.error(msg=eri)
        return {"Error": "Integrity Error"}
    except DatabaseError as erd:
        logger.error(msg=erd)
        return {"Error": "Database Error"}
    except Exception as e:
        logger.error(msg=e)
        return {"Error": e}

    ref_img = reference_img
    com_img = compare_img

    ref_img = cv2.imread(os.path.join(file_folder, 'referenceImage.jpg'))
    if (ref_img.shape[2] == 4):
        ref_img = ref_img[..., :3]

    com_img = cv2.imread(os.path.join(file_folder, 'compareImage.jpg'))
    if (com_img.shape[2] == 4):
        com_img = com_img[..., :3]

    ref_img_face, ref_img_bb = get_face(img=ref_img, pnet=pnet,
                                        rnet=rnet, onet=onet,
                                        image_size=image_size)
    if not ref_img_face:
        return {"result": [str(filename.split('.')[0]), "No Face in reference image"]}
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
            return {"result": [str(filename.split('.')[0]), id_name]}
        else:
            file_form.similarwith = "None"
            file_form.save()
            return {"result": [str(filename.split('.')[0]), "None"]}

    except Exception as e:
        logger.error(msg=e)
        return {"Error": e}


def object_detect(input_file, filename):
    """     Detecting Objects in image
    Args:
            *   input_file: Contents of the input image file
            *   filename: filename of the image
    Workflow:
            *   A numpy array of an image is taken as input (RGB).
            *   inference input dimension requires dimension of (416,416)
                therefore the input is first resizing to required
                input dimension and then normalized.
            *   Now the processed output is further processed to make it a
                json format which is compatible to TensorFlow Serving input.
            *   Then a http post request is made at localhost:8501.
                The post request contain data and headers.
            *   Incase of any exception, it return relevant error message.
            *   A list is maintained with each element being a dictionary
                with Label, Score, and Box being the keys and the name of the
                object, it's confidence score and it's bounding box
                coordinates as the respective values of these keys.
            *   A dictionary is returned with Objects as key and the list
                generated above as the value
    Returns:
            *   Dictionary having Objects as Key and list of dictionaries
                as the value where the dictionary element has Label, Score
                and Box as the keys and the name of the object, it's
                confidence score and it's bounding box coordinates as the
                respective values of these keys.
    """

    logger.info(msg="object_detect called")
    file_path = os.path.join(MEDIA_ROOT, 'object', filename)
    handle_uploaded_file(input_file, file_path)
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, tuple((416, 416)), interpolation=cv2.INTER_LINEAR)
    image = np.array(image, np.float32) / 255
    data = json.dumps({"inputs": [image.tolist()]})
    try:
        headers = {"content-type": "application/json"}
        url = urllib.parse.urljoin(base_url, object_detect_url)
        json_response = requests.post(url, data=data, headers=headers)
    except requests.exceptions.HTTPError as errh:
        logger.error(msg=errh)
        return {"Error": "An HTTP error occurred."}
    except requests.exceptions.ConnectionError as errc:
        logger.error(msg=errc)
        return {"Error": "A Connection error occurred."}
    except requests.exceptions.Timeout as errt:
        logger.error(msg=errt)
        return {"Error": "The request timed out."}
    except requests.exceptions.TooManyRedirects as errm:
        logger.error(msg=errm)
        return {"Error": "Bad URL"}
    except requests.exceptions.RequestException as err:
        logger.error(msg=err)
        return {"Error": "Facial Expression Recognition Not Working"}
    except Exception as e:
        logger.error(msg=e)
        return {"Error": "Facial Expression Recognition Not Working"}
    predictions = json.loads(json_response.text).get("outputs", "Bad request made.")
    boxes, scores, classes, nums = predictions["yolo_nms"][0], predictions[
        "yolo_nms_1"][0], predictions["yolo_nms_2"][0], predictions["yolo_nms_3"][0]
    result = []
    class_names = get_class_names(coco_names_path)
    for num in range(nums):
        result.append([{"Box": boxes[num]}, {"Score": scores[num]}, {"Label": class_names[int(classes[num])]}])
    return {"Objects": result}


def object_detect_video(input_file, filename):
    """     Detecting Objects in video
    Args:
            *   input_file: Contents of the input video file
            *   filename: filename of the video
    Workflow:
            *   uploaded file is read using opencv and gets processed
                frame by frame
            *   inference input dimension requires dimension of (416,416)
                therefore the frame is first resizing to required
                input dimension and then normalized.
            *   Now the processed output is further processed to make it a
                json format which is compatible to TensorFlow Serving input.
            *   Then a http post request is made at localhost:8501.
                The post request contain data and headers.
            *   Incase of any exception, it return relevant error message.
            *   A list is maintained with each element being a dictionary
                with Label, Score, and Box being the keys and the name of the
                object, it's confidence score and it's bounding box
                coordinates as the respective values of these keys.
            *   Result of every frame is stored in another list
            *   A dictionary is returned with Objects as key and the list
                generated above as the value
    Returns:
            *   Dictionary having Objects as Key and list of dictionaries
                as the value where the dictionary element has Label, Score
                and Box as the keys and the name of the object, it's
                confidence score and it's bounding box coordinates as the
                respective values of these keys for every frame
    """

    logger.info(msg="object_detect_video called")
    file_path = os.path.join(MEDIA_ROOT, 'object', filename)
    handle_uploaded_file(input_file, file_path)
    video_result = []
    vid = cv2.VideoCapture(file_path)
    while(vid.isOpened()):
        ret, image = vid.read()
        if ret:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, tuple((416, 416)), interpolation=cv2.INTER_LINEAR)
            image = np.array(image, np.float32) / 255
            data = json.dumps({"inputs": [image.tolist()]})
            try:
                headers = {"content-type": "application/json"}
                url = urllib.parse.urljoin(base_url, object_detect_url)
                json_response = requests.post(url, data=data, headers=headers)
            except requests.exceptions.HTTPError as errh:
                logger.error(msg=errh)
                return {"Error": "An HTTP error occurred."}
            except requests.exceptions.ConnectionError as errc:
                logger.error(msg=errc)
                return {"Error": "A Connection error occurred."}
            except requests.exceptions.Timeout as errt:
                logger.error(msg=errt)
                return {"Error": "The request timed out."}
            except requests.exceptions.TooManyRedirects as errm:
                logger.error(msg=errm)
                return {"Error": "Bad URL"}
            except requests.exceptions.RequestException as err:
                logger.error(msg=err)
                return {"Error": "Facial Expression Recognition Not Working"}
            except Exception as e:
                logger.error(msg=e)
                return {"Error": "Facial Expression Recognition Not Working"}
            predictions = json.loads(json_response.text).get("outputs", "Bad request made.")
            boxes, scores, classes, nums = predictions["yolo_nms"][0], predictions[
                "yolo_nms_1"][0], predictions["yolo_nms_2"][0], predictions["yolo_nms_3"][0]
            result = []
            class_names = get_class_names(coco_names_path)
            for num in range(nums):
                result.append([{"Box": boxes[num]}, {"Score": scores[num]}, {"Label": class_names[int(classes[num])]}])
            video_result.append(result)
        else:
            break
    return {"Objects": video_result}
