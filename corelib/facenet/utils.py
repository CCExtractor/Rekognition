import os
import numpy as np
import glob
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
from corelib.facenet.facenet import get_model_filenames
from corelib.facenet.align import detect_face
from corelib.facenet.facenet import load_img
# from scipy.misc import imsave
from skimage.io import imsave
import skimage
from collections import defaultdict
import string
import random
from Rekognition.settings import MEDIA_ROOT
from logger.logging import RekogntionLogger

tf.disable_v2_behavior()
logger = RekogntionLogger(name="utils")


def allowed_file(filename, allowed_set):
    """ to check whether the file being uploaded is a valid file or not
    Args:
        filename: filename of the file
        allowed_set: all the supported image format as set
    Returns:
            boolean value
    """

    logger.info(msg="allowed_file called")
    return "." in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


def remove_file_extension(filename):
    """ to remove the file extension

    Args:
        filename: filename of the file

        Returns:
            string (filename without extension)
    """

    logger.info(msg="remove_file_extension called")
    return os.path.splitext(filename)[0]


def save_image(img, filename, upload_path):
    """ to save the image

    Args:
        img: image file
        filename: filename of the file
        upload_path: path to upload

        Returns:
    """

    logger.info(msg="save_image called")
    try:
        imsave(os.path.join(upload_path, filename), arr=np.squeeze(img))
    except Exception as e:
        logger.error(msg=e)


def load_model(model):
    """to load the deep learning model

    Args:
        model: model path
    Returns:
        tensorflow graph

    """

    logger.info(msg="load_model called")
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name='')
            return graph
    else:
        meta_file, ckpt_file = get_model_filenames(model_exp)
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        graph = saver.restore(tf.get_default_session(),
                              os.path.join(model_exp, ckpt_file))
        return graph


def get_face(img, pnet, rnet, onet, image_size):
    """ to get the face from the image

    Args:
        img: image file
    Returns:
        numpy arrays of faces and corresponding faces

    """

    logger.info(msg="get_face called")
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    margin = 44
    input_image_size = image_size
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img=img, minsize=minsize,
                                                pnet=pnet, rnet=rnet,
                                                onet=onet,
                                                threshold=threshold,
                                                factor=factor)
    all_faces = []
    all_bb = []

    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            det = np.squeeze(face[0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]: bb[3], bb[0]:bb[2], :]
            # face_img = imresize(arr=cropped, size=(
            #     input_image_size, input_image_size), mode='RGB')
            face_img = skimage.transform.resize(cropped, (input_image_size, input_image_size))
            all_faces.append(face_img)
            all_bb.append(bb)
    return all_faces, all_bb


def embed_image(img, session, images_placeholder, phase_train_placeholder, embeddings, image_size):

    logger.info(msg="embed_image called")
    if img is not None:
        image = load_img(img=img, do_random_crop=False,
                         do_random_flip=False, do_prewhiten=True,
                         image_size=image_size)
        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
        embedding = session.run(embeddings, feed_dict=feed_dict)
        return embedding
    else:
        logger.error(msg="Embedding not generated as image is corrupt")
        return None


def save_face(img, where, filename):

    logger.info(msg="save_face called")
    path = os.path.join(MEDIA_ROOT, where, str(filename) + '.jpg')
    try:
        imsave(path, arr=np.squeeze(img))
    except Exception as e:
        logger.error(msg=e)


def save_embedding(embedding, filename, embeddings_path):

    logger.info(msg="save_embedding called")
    path = os.path.join(embeddings_path, str(filename))
    try:
        np.save(path, embedding)
    except Exception as e:
        logger.error(msg=e)


def load_embeddings(embeddings_path):

    logger.info(msg="load_embeddings called")
    embedding_dict = defaultdict()
    pathname = embeddings_path

    for embedding in glob.iglob(pathname=pathname + '/*.npy'):
        name = remove_file_extension(embedding).split('/')[-1]
        dict_embedding = np.load(embedding)
        embedding_dict[name] = dict_embedding

    return embedding_dict


def identify_face(embedding, embedding_dict):

    logger.info(msg="identify_face called")
    min_dis = 100
    identity = ''
    try:
        for(name, dict_embedding) in embedding_dict.items():
            distance = np.linalg.norm(embedding - dict_embedding)
            if distance < min_dis:
                min_dis = distance
                identity = name
        if min_dis <= 1.1:  # + ", the distance is " + str(min_dis)
            return identity
        else:
            result = "Unknown"
            return result
    except Exception as e:
        logger.error(msg=e)


def id_generator(size=30, chars=string.ascii_lowercase + string.digits):

    logger.info(msg="id_generator called")
    return ''.join(random.choice(chars) for _ in range(size))


def time_dura(dict_data, gap):

    logger.info(msg="time_dura called")
    result = {}
    new_list = []
    for name in dict_data:
        result[name] = []
        x, y = 0, 1
        z = 0
        for i in range(len(dict_data[name])):
            try:
                pre_time_stamp = round(dict_data[name][x], 2)
                post_time_stamp = round(dict_data[name][y], 2)
                if(abs(post_time_stamp - pre_time_stamp) > gap):
                    new_list.append((round(dict_data[name][z] / 1000, 2),
                                     round(dict_data[name][x] / 1000, 2)))
                    z = x + 1
                    x = y
                    y += 1
                else:
                    x += 1
                    y += 1
                if(x >= len(dict_data[name]) or y >= len(dict_data[name])):
                    break
            except Exception as e:
                logger.error(msg=e)

        new_list.append((round(dict_data[name][z] / 1000, 2),
                         round(dict_data[name][-1] / 1000, 2)))

        for xx in new_list:
            result[name].append((xx[0], xx[1]))
        new_list = []
    return result


def handle_uploaded_file(file, fname):

    logger.info(msg="handle_uploaded_file called")
    with open(fname, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)


def getnewuniquefilename(request):

    logger.info(msg="getnewuniquefilename called")
    file_ext = str((request.FILES['file'].name)).split('.')[-1]
    filename = id_generator() + '.' + file_ext
    return filename


def saveimageinfotodb(request, filename):

    logger.info(msg="saveimageinfotodb called")
    logger.warn(msg="image not saved to db")


def img_standardize(img):

    logger.info(msg="img_standardize called")
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img
