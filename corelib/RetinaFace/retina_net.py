import cv2
import os
import numpy as np
import json
from scipy.misc import imresize, imsave
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def retry_request(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class FaceDetectionRetina(object):
    """
    Used for FaceDetectionRetina, also this class acts as a sub mobule for
    embeddings and Video Recognition Modules
    """

    def __init__(self, down_scale_factor=1.0):

        self.down_scale_factor = down_scale_factor

    @staticmethod
    def _pad_input_image(img, max_steps):
        """pad image to suitable shape"""
        img_h, img_w, _ = img.shape

        img_pad_h = 0
        if img_h % max_steps > 0:
            img_pad_h = max_steps - img_h % max_steps

        img_pad_w = 0
        if img_w % max_steps > 0:
            img_pad_w = max_steps - img_w % max_steps

        padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
        img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                                 cv2.BORDER_CONSTANT, value=padd_val.tolist())
        pad_params = [img_h, img_w, img_pad_h, img_pad_w]

        return img, pad_params

    @staticmethod
    def _recover_pad_output(outputs, pad_params):
        """recover the padded output effect"""
        img_h, img_w, img_pad_h, img_pad_w = pad_params
        recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
            [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
        outputs[:, :14] = np.reshape(recover_xy, [-1, 14])
        return outputs

    @staticmethod
    def _draw_bbox_landm(img, ann, img_height, img_width):
        """draw bboxes and landmarks"""
        # bbox
        x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
            int(ann[2] * img_width), int(ann[3] * img_height)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # confidence
        text = "{:.4f}".format(ann[15])
        cv2.putText(img, text,
                    (int(ann[0] * img_width), int(ann[1] * img_height)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landmark
        if ann[14] > 0:
            cv2.circle(img, (int(ann[4] * img_width),
                             int(ann[5] * img_height)), 1, (255, 255, 0), 2)
            cv2.circle(img, (int(ann[6] * img_width),
                             int(ann[7] * img_height)), 1, (0, 255, 255), 2)
            cv2.circle(img, (int(ann[8] * img_width),
                             int(ann[9] * img_height)), 1, (255, 0, 0), 2)
            cv2.circle(img, (int(ann[10] * img_width),
                             int(ann[11] * img_height)), 1, (0, 100, 255), 2)
            cv2.circle(img, (int(ann[12] * img_width),
                             int(ann[13] * img_height)), 1, (255, 0, 100), 2)

    @staticmethod
    def _draw_anchor(img, prior, img_height, img_width):
        """draw anchors"""
        x1 = int(prior[0] * img_width - prior[2] * img_width / 2)
        y1 = int(prior[1] * img_height - prior[3] * img_height / 2)
        x2 = int(prior[0] * img_width + prior[2] * img_width / 2)
        y2 = int(prior[1] * img_height + prior[3] * img_height / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)

    @staticmethod
    def _process_outputs(output, img_height_raw, img_width_raw):
        return {
            "Bounding Boxes": {
                "confidance": output[15],
                "x1": output[0] * img_width_raw,
                "y1": output[1] * img_height_raw,
                "x2": output[2] * img_width_raw,
                "y2": output[3] * img_height_raw,
            },
            "Landmarks": {
                "confidance": output[14],
                "eye_left": (output[4] * img_width_raw, output[5] * img_height_raw),
                "eye_right": (output[6] * img_width_raw, output[7] * img_height_raw),
                "nose": (output[8] * img_width_raw, output[9] * img_height_raw),
                "mouth_left": (output[10] * img_width_raw, output[11] * img_height_raw),
                "mouth_right": (output[12] * img_width_raw, output[13] * img_height_raw),

            }
        }

    def get_face(self, file):
        """ to get the face from the image

        Args:
            img: image file
        Returns:
            numpy arrays of faces and corresponding faces

        """
        size = (160, 160)
        img_raw = cv2.imread(file)
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())

        prediction, _ = self.predict(img)
        print("Number of Faces from Predict " + str(len(prediction)))
        all_faces = []
        all_bb = []

        if not len(prediction) == 0:
            for face in prediction:
                face = face["Bounding Boxes"]

                bb = np.zeros(4, dtype=np.int32)
                bb[0] = int(face["x1"])
                bb[1] = int(face["y1"])
                bb[2] = int(face["x2"])
                bb[3] = int(face["y2"])
                cropped = img[bb[1]:bb[3], bb[0]: bb[2], :]
                face_img = imresize(arr=cropped, size=size, mode='RGB')
                all_faces.append(face_img)
                all_bb.append(bb)
        return all_faces, all_bb

    def predict(self, img):
        """  Returns bounding box coordinates along with resultant image  """
        img_height_raw, img_width_raw, _ = img.shape
        if self.down_scale_factor < 1.0:
            img = cv2.resize(img, (0, 0), fx=self.down_scale_factor,
                             fy=self.down_scale_factor,
                             interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img, pad_params = self._pad_input_image(
            img, max_steps=max([8, 16, 32]))

        headers = {"content-type": "application/json"}

        data = json.dumps({"signature_name": "serving_default",
                           "inputs": img[np.newaxis, ...].tolist()})

        json_response = retry_request().post(
            'http://localhost:8501/v1/models/retinanet:predict',
            data=data, headers=headers)

        outputs = json.loads(json_response.text)["outputs"]
        outputs = np.array(outputs)

        outputs = self._recover_pad_output(outputs, pad_params)
        processed_outputs = []
        for prior_index in range(len(outputs)):
            self._draw_bbox_landm(
                img, outputs[prior_index], img_height_raw, img_width_raw)
            processed_outputs.append(
                self._process_outputs(outputs[prior_index],
                                      img_height_raw, img_width_raw))
        return processed_outputs, img
