from Rekognition.settings import BASE_DIR
from corelib.facenet.utils import load_model, load_embeddings
from corelib.facenet.align import detect_face
import tensorflow as tf
import os


upload_path = os.path.join(BASE_DIR, 'cceface/uploads')
embeddings_path = os.path.join(BASE_DIR, 'corelib/embeddings')
allowed_set = set(['png', 'jpg', 'jpeg', 'PNG', 'JPEG', 'JPG'])
facenet_model_path = BASE_DIR + '/corelib/model/facenet/2017/20170512-110547.pb'
facenet_model = load_model(facenet_model_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
image_size = 160
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
facenet_persistent_session = tf.Session(graph=facenet_model, config=config)
pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)
embedding_dict = load_embeddings(embeddings_path)

Facial_expression_class_names = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]  # Don't change the order

nsfw_class_names = {0: 'Drawings', 1: 'Hentai', 2: 'Neutral', 3: 'Porn', 4: 'Sexy'}
