import numpy as np
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D
import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import cv2
from numpy.linalg import norm

eps = 1e-10


def polygon_to_rbox2(xy):
    # two points at the top left and top right corner plus height
    tl, tr, br, bl = xy
    # length of top and bottom edge
    dt, _ = tr - tl, bl - br
    # height is mean between distance from top to bottom right and distance from top edge to bottom left
    h = (norm(np.cross(dt, tl - br)) + norm(np.cross(dt, tr - bl))) / (2 * (norm(dt) + eps))
    return np.hstack((tl, tr, h))


def rbox2_to_polygon(rbox):
    x1, y1, x2, y2, h = rbox
    alpha = np.arctan2(x1 - x2, y2 - y1)
    dx = -h * np.cos(alpha)
    dy = -h * np.sin(alpha)
    xy = np.reshape([x1, y1, x2, y2, x2 + dx, y2 + dy, x1 + dx, y1 + dy], (-1, 2))
    return xy


def polygon_to_rbox3(xy):
    # two points at the center of the left and right edge plus heigth
    tl, tr, br, bl = xy
    # length of top and bottom edge
    dt, _ = tr - tl, bl - br
    # height is mean between distance from top to bottom right and distance from top edge to bottom left
    h = (norm(np.cross(dt, tl - br)) + norm(np.cross(dt, tr - bl))) / (2 * (norm(dt) + eps))
    p1 = (tl + bl) / 2.
    p2 = (tr + br) / 2.
    return np.hstack((p1, p2, h))


def rbox3_to_polygon(rbox):
    x1, y1, x2, y2, h = rbox
    alpha = np.arctan2(x1 - x2, y2 - y1)
    dx = -h * np.cos(alpha) / 2.
    dy = -h * np.sin(alpha) / 2.
    xy = np.reshape([x1 - dx, y1 - dy, x2 - dx, y2 - dy, x2 + dx, y2 + dy, x1 + dx, y1 + dy], (-1, 2))
    return xy


def polygon_to_box(xy, box_format='xywh'):
    # minimum axis aligned bounding box containing some points
    xy = np.reshape(xy, (-1, 2))
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    if box_format == 'xywh':
        box = [xmin, ymin, xmax - xmin, ymax - ymin]
    elif box_format == 'xyxy':
        box = [xmin, ymin, xmax, ymax]
    if box_format == 'polygon':
        box = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
    return np.array(box)


def non_maximum_suppression_slow(boxes, confs, iou_threshold, top_k):
    """Does None-Maximum Suppresion on detection results.

    Intuitive but slow as hell!!!

    # Agruments
        boxes: Array of bounding boxes (boxes, xmin + ymin + xmax + ymax).
        confs: Array of corresponding confidenc values.
        iou_threshold: Intersection over union threshold used for comparing
            overlapping boxes.
        top_k: Maximum number of returned indices.

    # Return
        List of remaining indices.
    """
    idxs = np.argsort(-confs)
    selected = []
    for idx in idxs:
        if np.any(iou(boxes[idx], boxes[selected]) >= iou_threshold):
            continue
        selected.append(idx)
        if len(selected) >= top_k:
            break
    return selected


def non_maximum_suppression(boxes, confs, overlap_threshold, top_k):
    """Does None-Maximum Suppresion on detection results.

    # Agruments
        boxes: Array of bounding boxes (boxes, xmin + ymin + xmax + ymax).
        confs: Array of corresponding confidenc values.
        overlap_threshold:
        top_k: Maximum number of returned indices.

    # Return
        List of remaining indices.

    # References
        - Girshick, R. B. and Felzenszwalb, P. F. and McAllester, D.
          [Discriminatively Trained Deformable Part Models, Release 5](http://people.cs.uchicago.edu/~rbg/latent-release5/)
    """
    eps = 1e-15

    boxes = np.asarray(boxes, dtype='float32')

    pick = []
    x1, y1, x2, y2 = boxes.T

    idxs = np.argsort(confs)
    area = (x2 - x1) * (y2 - y1)

    while len(idxs) > 0:
        i = idxs[-1]

        pick.append(i)
        if len(pick) >= top_k:
            break

        idxs = idxs[:-1]

        xx1 = np.maximum(x1[i], x1[idxs])
        yy1 = np.maximum(y1[i], y1[idxs])
        xx2 = np.minimum(x2[i], x2[idxs])
        yy2 = np.minimum(y2[i], y2[idxs])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        I = w * h

        overlap = I / (area[idxs] + eps)
        # as in Girshick et. al.

        #U = area[idxs] + area[i] - I
        #overlap = I / (U + eps)

        idxs = idxs[overlap <= overlap_threshold]

    return pick


def bn_acti_conv(x, filters, kernel_size=1, stride=1, padding='same', activation='relu'):
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    if kernel_size > 1:
        x = SeparableConv2D(filters, kernel_size, depth_multiplier=1, strides=stride, padding=padding)(x)
    else:
        x = Conv2D(filters, kernel_size, strides=stride, padding=padding)(x)
    return x


def dense_block(x, n, growth_rate, width=4, activation='relu'):
    input_shape = K.int_shape(x)
    c = input_shape[3]
    for i in range(n):
        x1 = x
        x2 = bn_acti_conv(x, growth_rate * width, 1, 1, activation=activation)
        x2 = bn_acti_conv(x2, growth_rate, 3, 1, activation=activation)
        x = concatenate([x1, x2], axis=3)
        c += growth_rate
    return x


def downsampling_block(x, filters, width, padding='same', activation='relu'):
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x1 = MaxPooling2D(pool_size=2, strides=2, padding=padding)(x)
    x2 = DepthwiseConv2D(3, depth_multiplier=1, strides=2, padding=padding)(x)
    x = concatenate([x1, x2], axis=3)
    x = Conv2D(filters, 1, strides=1)(x)
    return x


def ssd512_dense_separable_body(x, activation='relu'):
    # used for SegLink and TextBoxes++ variantes with separable convolution

    if activation == 'leaky_relu':
        activation = leaky_relu

    growth_rate = 48
    compressed_features = 224
    source_layers = []

    x = SeparableConv2D(96, 3, depth_multiplier=32, strides=2, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = SeparableConv2D(96, 3, depth_multiplier=1, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = SeparableConv2D(96, 3, depth_multiplier=1, strides=1, padding='same')(x)
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)

    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = dense_block(x, 6, growth_rate, 4, activation)
    x = bn_acti_conv(x, compressed_features, 1, 1, activation=activation)

    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    x = dense_block(x, 6, growth_rate, 4, activation)
    x = bn_acti_conv(x, compressed_features, 1, 1, activation=activation)
    source_layers.append(x)  # 64x64

    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = dense_block(x, 6, growth_rate, 4, activation)
    x = bn_acti_conv(x, compressed_features, 1, 1, activation=activation)
    source_layers.append(x)  # 32x32

    x = downsampling_block(x, 192, 1, activation=activation)
    source_layers.append(x)  # 16x16

    x = downsampling_block(x, 160, 1, activation=activation)
    source_layers.append(x)  # 8x8

    x = downsampling_block(x, 128, 1, activation=activation)
    source_layers.append(x)  # 4x4

    x = downsampling_block(x, 96, 1, activation=activation)
    source_layers.append(x)  # 2x2

    x = downsampling_block(x, 64, 1, activation=activation)
    source_layers.append(x)  # 1x1

    return source_layers


def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def leaky_relu(x):
    """Leaky Rectified Linear activation.

    # References
        - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
    """
    # return K.relu(x, alpha=0.1, max_value=None)

    # requires less memory than keras implementation
    alpha = 0.1
    zero = _to_tensor(0., x.dtype.base_dtype)
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x = alpha * tf.minimum(x, zero) + tf.maximum(x, zero)
    return x


class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Default feature scale.

    # Input shape
        4D tensor with shape: (samples, rows, cols, channels)

    # Output shape
        Same as input

    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf

    # TODO
        Add possibility to have one scale for all features.
    """

    def __init__(self, scale=20, **kwargs):
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name=self.name + '_gamma',
                                     shape=(input_shape[-1],),
                                     initializer=initializers.Constant(self.scale),
                                     trainable=True)
        super(Normalize, self).build(input_shape)

    def call(self, x, mask=None):
        return self.gamma * K.l2_normalize(x, axis=-1)


def rot_matrix(theta):
    s, c = np.sin(theta), np.cos(theta)
    return np.array([[c, -s], [s, c]])


def polygon_to_rbox(xy):
    # center point plus width, height and orientation angle
    tl, tr, br, bl = xy
    # length of top and bottom edge
    dt, db = tr - tl, bl - br
    # center is mean of all 4 vetrices
    cx, cy = c = np.sum(xy, axis=0) / len(xy)
    # width is mean of top and bottom edge length
    w = (norm(dt) + norm(db)) / 2.
    # height is distance from center to top edge plus distance form center to bottom edge
    h = norm(np.cross(dt, tl - c)) / (norm(dt) + eps) + norm(np.cross(db, br - c)) / (norm(db) + eps)
    #h = point_line_distance(c, tl, tr) +  point_line_distance(c, br, bl)
    #h = (norm(tl-bl) + norm(tr-br)) / 2.
    # angle is mean of top and bottom edge angle
    theta = (np.arctan2(dt[0], dt[1]) + np.arctan2(db[0], db[1])) / 2.
    return np.array([cx, cy, w, h, theta])


def rbox_to_polygon(rbox):
    cx, cy, w, h, theta = rbox
    box = np.array([[-w, h], [w, h], [w, -h], [-w, -h]]) / 2.
    box = np.dot(box, rot_matrix(theta))
    box += rbox[:2]
    return box


def iou(box, boxes):
    """Computes the intersection over union for a given axis
    aligned bounding box with several others.

    # Arguments
        box: Bounding box, numpy array of shape (4).
            (x1, y1, x2, y2)
        boxes: Reference bounding boxes, numpy array of
            shape (num_boxes, 4).

    # Return
        iou: Intersection over union,
            numpy array of shape (num_boxes).
    """
    # compute intersection
    inter_upleft = np.maximum(boxes[:, :2], box[:2])
    inter_botright = np.minimum(boxes[:, 2:4], box[2:])
    inter_wh = inter_botright - inter_upleft
    inter_wh = np.maximum(inter_wh, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    # compute union
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_pred + area_gt - inter
    # compute iou
    iou = inter / union
    return iou


class PriorMap(object):
    """Handles prior boxes for a given feature map.

    # Arguments / Attributes
        source_layer_name
        image_size: Tuple with spatial size of model input.
        map_size
        variances
        aspect_ratios: List of aspect ratios for the prior boxes at each
            location.
        shift: List of tuples for the displacement of the prior boxes
            relative to ther location. Each tuple contains an value between
            -1.0 and 1.0 for x and y direction.
        clip: Boolean, whether the boxes should be cropped to do not exceed
            the borders of the input image.
        step
        minmax_size: List of tuples with s_min and s_max values (see paper).
        special_ssd_box: Boolean, wether or not the extra box for aspect
            ratio 1 is used.

    # Notes
        The compute_priors methode has to be called to get usable prior boxes.
    """

    def __init__(self, source_layer_name, image_size, map_size,
                 minmax_size=None, variances=[0.1, 0.1, 0.2, 0.2],
                 aspect_ratios=[1], shift=None,
                 clip=False, step=None, special_ssd_box=False):

        self.__dict__.update(locals())

        # self.compute_priors()

    def __str__(self):
        s = ''
        for a in ['source_layer_name',
                  'map_size',
                  'aspect_ratios',
                  'shift',
                  'clip',
                  'minmax_size',
                  'special_ssd_box',
                  'num_locations',
                  'num_boxes',
                  'num_boxes_per_location',
                  ]:
            s += '%-24s %s\n' % (a, getattr(self, a))
        return s

    @property
    def num_boxes_per_location(self):
        return len(self.box_wh)

    @property
    def num_locations(self):
        return len(self.box_xy)

    @property
    def num_boxes(self):
        return len(self.box_xy) * len(self.box_wh)  # len(self.priors)

    def compute_priors(self):
        image_h, image_w = image_size = self.image_size
        map_h, map_w = map_size = self.map_size
        min_size, max_size = self.minmax_size

        # define centers of prior boxes
        if self.step is None:
            step_x = image_w / map_w
            step_y = image_h / map_h
            assert step_x % 1 == 0 and step_y % 1 == 0, 'map size %s not constiten with input size %s' % (map_size, image_size)
        else:
            step_x = step_y = self.step

        linx = np.array([(0.5 + i) for i in range(map_w)]) * step_x
        liny = np.array([(0.5 + i) for i in range(map_h)]) * step_y
        box_xy = np.array(np.meshgrid(linx, liny)).reshape(2, -1).T

        if self.shift is None:
            shift = [(0.0, 0.0)] * len(self.aspect_ratios)
        else:
            shift = self.shift

        box_wh = []
        box_shift = []
        for i in range(len(self.aspect_ratios)):
            ar = self.aspect_ratios[i]
            box_wh.append([min_size * np.sqrt(ar), min_size / np.sqrt(ar)])
            box_shift.append(shift[i])
            if ar == 1 and self.special_ssd_box:  # special SSD box
                box_wh.append([np.sqrt(min_size * max_size), np.sqrt(min_size * max_size)])
                box_shift.append((0.0, 0.0))
        box_wh = np.asarray(box_wh)

        box_shift = np.asarray(box_shift)
        box_shift = np.clip(box_shift, -1.0, 1.0)
        box_shift = box_shift * np.array([step_x, step_y])  # percent to pixels

        # values for individual prior boxes
        priors_shift = np.tile(box_shift, (len(box_xy), 1))
        priors_xy = np.repeat(box_xy, len(box_wh), axis=0) + priors_shift
        priors_wh = np.tile(box_wh, (len(box_xy), 1))

        priors_min_xy = priors_xy - priors_wh / 2.
        priors_max_xy = priors_xy + priors_wh / 2.

        if self.clip:
            priors_min_xy[:, 0] = np.clip(priors_min_xy[:, 0], 0, image_w)
            priors_min_xy[:, 1] = np.clip(priors_min_xy[:, 1], 0, image_h)
            priors_max_xy[:, 0] = np.clip(priors_max_xy[:, 0], 0, image_w)
            priors_max_xy[:, 1] = np.clip(priors_max_xy[:, 1], 0, image_h)

        priors_variances = np.tile(self.variances, (len(priors_xy), 1))

        self.box_xy = box_xy
        self.box_wh = box_wh
        self.box_shfit = box_shift

        self.priors_xy = priors_xy
        self.priors_wh = priors_wh
        self.priors_min_xy = priors_min_xy
        self.priors_max_xy = priors_max_xy
        self.priors_variances = priors_variances
        self.priors = np.concatenate([priors_min_xy, priors_max_xy, priors_variances], axis=1)

    def plot_locations(self, color='r'):
        xy = self.box_xy
        plt.plot(xy[:, 0], xy[:, 1], '.', color=color, markersize=6)

    def plot_boxes(self, location_idxs=[]):
        colors = 'rgbcmy'
        ax = plt.gca()
        n = self.num_boxes_per_location
        for i in location_idxs:
            for j in range(n):
                idx = i * n + j
                if idx >= self.num_boxes:
                    break
                x1, y1, x2, y2 = self.priors[idx, :4]
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor=colors[j % len(colors)], linewidth=2))
        ax.autoscale_view()


class SSDPriorUtil(object):
    """Utility for SSD prior boxes.
    """

    def __init__(self, model, aspect_ratios=None, shifts=None,
                 minmax_sizes=None, steps=None, scale=None, clips=None,
                 special_ssd_boxes=None, ssd_assignment=None):

        source_layers_names = [l.name.split('/')[0] for l in model.source_layers]
        self.source_layers_names = source_layers_names

        self.model = model
        self.image_size = model.input_shape[1:3]

        num_maps = len(source_layers_names)

        # take parameters from model definition if they exist there
        if aspect_ratios is None:
            if hasattr(model, 'aspect_ratios'):
                aspect_ratios = model.aspect_ratios
            else:
                aspect_ratios = [[1]] * num_maps

        if shifts is None:
            if hasattr(model, 'shifts'):
                shifts = model.shifts
            else:
                shifts = [None] * num_maps

        if minmax_sizes is None:
            if hasattr(model, 'minmax_sizes'):
                minmax_sizes = model.minmax_sizes
            else:
                # as in equation (4)
                min_dim = np.min(self.image_size)
                min_ratio = 10  # 15
                max_ratio = 100  # 90
                s = np.linspace(min_ratio, max_ratio, num_maps + 1) * min_dim / 100.
                minmax_sizes = [(round(s[i]), round(s[i + 1])) for i in range(len(s) - 1)]

        if scale is None:
            if hasattr(model, 'scale'):
                scale = model.scale
            else:
                scale = 1.0
        minmax_sizes = np.array(minmax_sizes) * scale

        if steps is None:
            if hasattr(model, 'steps'):
                steps = model.steps
            else:
                steps = [None] * num_maps

        if clips is None:
            if hasattr(model, 'clips'):
                clips = model.clips
            else:
                clips = False
        if isinstance(clips, bool):
            clips = [clips] * num_maps

        if special_ssd_boxes is None:
            if hasattr(model, 'special_ssd_boxes'):
                special_ssd_boxes = model.special_ssd_boxes
            else:
                special_ssd_boxes = False
        if isinstance(special_ssd_boxes, bool):
            special_ssd_boxes = [special_ssd_boxes] * num_maps

        if ssd_assignment is None:
            if hasattr(model, 'ssd_assignment'):
                ssd_assignment = model.ssd_assignment
            else:
                ssd_assignment = True
        self.ssd_assignment = ssd_assignment

        self.prior_maps = []
        for i in range(num_maps):
            layer = model.get_layer(source_layers_names[i])
            map_h, map_w = map_size = layer.output_shape[1:3]
            m = PriorMap(source_layer_name=source_layers_names[i],
                         image_size=self.image_size,
                         map_size=map_size,
                         minmax_size=minmax_sizes[i],
                         variances=[0.1, 0.1, 0.2, 0.2],
                         aspect_ratios=aspect_ratios[i],
                         shift=shifts[i],
                         step=steps[i],
                         special_ssd_box=special_ssd_boxes[i],
                         clip=clips[i])
            self.prior_maps.append(m)
        self.update_priors()

        self.nms_top_k = 400
        self.nms_thresh = 0.45

    @property
    def num_maps(self):
        return len(self.prior_maps)

    def update_priors(self):
        priors_xy = []
        priors_wh = []
        priors_min_xy = []
        priors_max_xy = []
        priors_variances = []
        priors = []

        map_offsets = [0]
        for i in range(len(self.prior_maps)):
            m = self.prior_maps[i]

            # compute prior boxes
            m.compute_priors()

            # collect prior data
            priors_xy.append(m.priors_xy)
            priors_wh.append(m.priors_wh)
            priors_min_xy.append(m.priors_min_xy)
            priors_max_xy.append(m.priors_max_xy)
            priors_variances.append(m.priors_variances)
            priors.append(m.priors)
            map_offsets.append(map_offsets[-1] + len(m.priors))

        self.priors_xy = np.concatenate(priors_xy, axis=0)
        self.priors_wh = np.concatenate(priors_wh, axis=0)
        self.priors_min_xy = np.concatenate(priors_min_xy, axis=0)
        self.priors_max_xy = np.concatenate(priors_max_xy, axis=0)
        self.priors_variances = np.concatenate(priors_variances, axis=0)
        self.priors = np.concatenate(priors, axis=0)
        self.map_offsets = map_offsets

        # normalized prior boxes
        image_wh = self.image_size[::-1]
        self.priors_xy_norm = self.priors_xy / image_wh
        self.priors_wh_norm = self.priors_wh / image_wh
        self.priors_min_xy_norm = self.priors_min_xy / image_wh
        self.priors_max_xy_norm = self.priors_max_xy / image_wh
        self.priors_norm = np.concatenate([self.priors_min_xy_norm, self.priors_max_xy_norm, self.priors_variances], axis=1)

    def encode(self, gt_data, overlap_threshold=0.45, debug=False):
        # calculation is done with normalized sizes

        # TODO: empty ground truth
        if gt_data.shape[0] == 0:
            print('gt_data', type(gt_data), gt_data.shape)

        num_classes = self.model.num_classes
        num_priors = self.priors.shape[0]

        gt_boxes = self.gt_boxes = np.copy(gt_data[:, :4])  # normalized xmin, ymin, xmax, ymax
        gt_class_idx = np.asarray(gt_data[:, -1] + 0.5, dtype=np.int)
        gt_one_hot = np.zeros([len(gt_class_idx), num_classes])
        gt_one_hot[range(len(gt_one_hot)), gt_class_idx] = 1  # one_hot classes including background

        gt_min_xy = gt_boxes[:, 0:2]
        gt_max_xy = gt_boxes[:, 2:4]
        gt_xy = (gt_boxes[:, 2:4] + gt_boxes[:, 0:2]) / 2.
        gt_wh = gt_boxes[:, 2:4] - gt_boxes[:, 0:2]

        gt_iou = np.array([iou(b, self.priors_norm) for b in gt_boxes]).T
        max_idxs = np.argmax(gt_iou, axis=1)

        priors_xy = self.priors_xy_norm
        priors_wh = self.priors_wh_norm

        # assign ground truth to priors
        if self.ssd_assignment:
            # original ssd assignment rule
            max_idxs = np.argmax(gt_iou, axis=1)
            max_val = gt_iou[np.arange(num_priors), max_idxs]
            prior_mask = max_val > overlap_threshold
            match_indices = max_idxs[prior_mask]
        else:
            prior_area = np.product(priors_wh, axis=-1)[:, None]
            gt_area = np.product(gt_wh, axis=-1)[:, None]

            priors_ar = priors_wh[:, 0] / priors_wh[:, 1]
            gt_ar = gt_wh[:, 0] / gt_wh[:, 1]

            match_mask = np.array([np.concatenate([
                priors_xy >= gt_min_xy[i],
                priors_xy <= gt_max_xy[i],
                #priors_wh >= 0.5 * gt_wh[i],
                #priors_wh <= 2.0 * gt_wh[i],
                #prior_area >= 0.25 * gt_area[i],
                #prior_area <= 4.0 * gt_area[i],
                prior_area >= 0.0625 * gt_area[i],
                prior_area <= 1.0 * gt_area[i],
                #((priors_ar < 1.0) == (gt_ar[i] < 1.0))[:,None],
                (np.abs(priors_ar - gt_ar[i]) < 0.5)[:, None],
                max_idxs[:, None] == i
            ], axis=-1) for i in range(len(gt_boxes))])
            self.match_mask = match_mask
            match_mask = np.array([np.all(m, axis=-1) for m in match_mask]).T
            prior_mask = np.any(match_mask, axis=-1)
            match_indices = np.argmax(match_mask[prior_mask, :], axis=-1)

        self.match_indices = dict(zip(list(np.ix_(prior_mask)[0]), list(match_indices)))

        # prior labels
        confidence = np.zeros((num_priors, num_classes))
        confidence[:, 0] = 1
        confidence[prior_mask] = gt_one_hot[match_indices]

        # compute local offsets from ground truth boxes
        gt_xy = gt_xy[match_indices]
        gt_wh = gt_wh[match_indices]
        priors_xy = priors_xy[prior_mask]
        priors_wh = priors_wh[prior_mask]
        priors_variances = self.priors_variances[prior_mask, :]
        offsets = np.zeros((num_priors, 4))
        offsets[prior_mask, 0:2] = (gt_xy - priors_xy) / priors_wh
        offsets[prior_mask, 2:4] = np.log(gt_wh / priors_wh)
        offsets[prior_mask, 0:4] /= priors_variances

        return np.concatenate([offsets, confidence], axis=1)

    def decode(self, model_output, confidence_threshold=0.01, keep_top_k=200, fast_nms=True, sparse=True):
        # calculation is done with normalized sizes

        prior_mask = model_output[:, 4:] > confidence_threshold
        image_wh = self.image_size[::-1]

        if sparse:
            # compute boxes only if the confidence is high enough and the class is not background
            mask = np.any(prior_mask[:, 1:], axis=1)
            prior_mask = prior_mask[mask]
            mask = np.ix_(mask)[0]
            model_output = model_output[mask]
            priors_xy = self.priors_xy[mask] / image_wh
            priors_wh = self.priors_wh[mask] / image_wh
            priors_variances = self.priors_variances[mask, :]
        else:
            priors_xy = self.priors_xy / image_wh
            priors_wh = self.priors_wh / image_wh
            priors_variances = self.priors_variances

        offsets = model_output[:, :4]
        confidence = model_output[:, 4:]

        num_priors = offsets.shape[0]
        num_classes = confidence.shape[1]

        # compute bounding boxes from local offsets
        boxes = np.empty((num_priors, 4))
        offsets = offsets * priors_variances
        boxes_xy = priors_xy + offsets[:, 0:2] * priors_wh
        boxes_wh = priors_wh * np.exp(offsets[:, 2:4])
        boxes[:, 0:2] = boxes_xy - boxes_wh / 2.  # xmin, ymin
        boxes[:, 2:4] = boxes_xy + boxes_wh / 2.  # xmax, ymax
        boxes = np.clip(boxes, 0.0, 1.0)

        # do non maximum suppression
        results = []
        for c in range(1, num_classes):
            mask = prior_mask[:, c]
            boxes_to_process = boxes[mask]
            if len(boxes_to_process) > 0:
                confs_to_process = confidence[mask, c]

                if fast_nms:
                    idx = non_maximum_suppression(
                        boxes_to_process, confs_to_process,
                        self.nms_thresh, self.nms_top_k)
                else:
                    idx = non_maximum_suppression_slow(
                        boxes_to_process, confs_to_process,
                        self.nms_thresh, self.nms_top_k)

                good_boxes = boxes_to_process[idx]
                good_confs = confs_to_process[idx][:, None]
                labels = np.ones((len(idx), 1)) * c
                c_pred = np.concatenate((good_boxes, good_confs, labels), axis=1)
                results.extend(c_pred)
        if len(results) > 0:
            results = np.array(results)
            order = np.argsort(-results[:, 4])
            results = results[order]
            results = results[:keep_top_k]
        else:
            results = np.empty((0, 6))
        self.results = results
        return results

    def compute_class_weights(self, gt_util, num_samples=np.inf):
        """Computes weighting factors for the classification loss by considering
        the inverse frequency of class instance in local ground truth.
        """
        s = np.zeros(gt_util.num_classes)
        for i in tqdm(range(min(gt_util.num_samples, num_samples))):
            egt = self.encode(gt_util.data[i])
            s += np.sum(egt[:, -gt_util.num_classes:], axis=0)
        si = 1 / s
        return si / np.sum(si) * len(s)

    def show_image(self, img):
        """Resizes an image to the network input size and shows it in the current figure.
        """
        image_wh = self.image_size[::-1]
        img = cv2.resize(img, image_wh, cv2.INTER_LINEAR)
        img = img[:, :, (2, 1, 0)]  # BGR to RGB
        img = img / 256.
        plt.imshow(img)

    def plot_assignment(self, map_idx):
        ax = plt.gca()
        im = plt.gci()
        image_h, image_w = image_size = im.get_size()

        # ground truth
        boxes = self.gt_boxes
        boxes_x = (boxes[:, 0] + boxes[:, 2]) / 2. * image_w
        boxes_y = (boxes[:, 1] + boxes[:, 3]) / 2. * image_h
        for box in boxes:
            xy_rec = to_rec(box[:4], image_size)
            ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor='b', linewidth=2))
        plt.plot(boxes_x, boxes_y, 'bo', markersize=6)

        # prior boxes
        for idx, box_idx in self.match_indices.items():
            if idx >= self.map_offsets[map_idx] and idx < self.map_offsets[map_idx + 1]:
                x, y = self.priors_xy[idx]
                w, h = self.priors_wh[idx]
                plt.plot(x, y, 'ro', markersize=4)
                plt.plot([x, boxes_x[box_idx]], [y, boxes_y[box_idx]], '-r', linewidth=1)
                ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w + 1, h + 1,
                                           fill=False, edgecolor='y', linewidth=2))

    def plot_results(self, results=None, classes=None, show_labels=True, gt_data=None, confidence_threshold=None):
        if results is None:
            results = self.results
        if confidence_threshold is not None:
            mask = results[:, 4] > confidence_threshold
            results = results[mask]
        if classes is not None:
            colors = plt.cm.hsv(np.linspace(0, 1, len(classes) + 1)).tolist()
        ax = plt.gca()
        im = plt.gci()
        image_size = im.get_size()

        # draw ground truth
        if gt_data is not None:
            for box in gt_data:
                label = np.nonzero(box[4:])[0][0] + 1
                color = 'g' if classes is None else colors[label]
                xy_rec = to_rec(box[:4], image_size)
                ax.add_patch(plt.Polygon(xy_rec, fill=True, color=color, linewidth=1, alpha=0.3))

        # draw prediction
        for r in results:
            label = int(r[5])
            confidence = r[4]
            color = 'r' if classes is None else colors[label]
            xy_rec = to_rec(r[:4], image_size)
            ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor=color, linewidth=2))
            if show_labels:
                label_name = label if classes is None else classes[label]
                xmin, ymin = xy_rec[0]
                display_txt = '%0.2f, %s' % (confidence, label_name)
                ax.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    def print_gt_stats(self):
        # TODO
        pass


class PriorUtil(SSDPriorUtil):
    """Utility for SSD prior boxes.
    """

    def encode(self, gt_data, overlap_threshold=0.5, debug=False):
        # calculation is done with normalized sizes

        # TODO: empty ground truth
        if gt_data.shape[0] == 0:
            print('gt_data', type(gt_data), gt_data.shape)

        num_classes = 2
        num_priors = self.priors.shape[0]

        gt_polygons = np.copy(gt_data[:, :8])  # normalized quadrilaterals
        gt_rboxes = np.array([polygon_to_rbox3(np.reshape(p, (-1, 2))) for p in gt_data[:, :8]])

        # minimum horizontal bounding rectangles
        gt_xmin = np.min(gt_data[:, 0:8:2], axis=1)
        gt_ymin = np.min(gt_data[:, 1:8:2], axis=1)
        gt_xmax = np.max(gt_data[:, 0:8:2], axis=1)
        gt_ymax = np.max(gt_data[:, 1:8:2], axis=1)
        gt_boxes = self.gt_boxes = np.array([gt_xmin, gt_ymin, gt_xmax, gt_ymax]).T  # normalized xmin, ymin, xmax, ymax

        gt_class_idx = np.asarray(gt_data[:, -1] + 0.5, dtype=np.int)
        gt_one_hot = np.zeros([len(gt_class_idx), num_classes])
        gt_one_hot[range(len(gt_one_hot)), gt_class_idx] = 1  # one_hot classes including background

        gt_iou = np.array([iou(b, self.priors_norm) for b in gt_boxes]).T

        # assigne gt to priors
        max_idxs = np.argmax(gt_iou, axis=1)
        max_val = gt_iou[np.arange(num_priors), max_idxs]
        prior_mask = max_val > overlap_threshold
        match_indices = max_idxs[prior_mask]

        self.match_indices = dict(zip(list(np.ix_(prior_mask)[0]), list(match_indices)))

        # prior labels
        confidence = np.zeros((num_priors, num_classes))
        confidence[:, 0] = 1
        confidence[prior_mask] = gt_one_hot[match_indices]

        gt_xy = (gt_boxes[:, 2:4] + gt_boxes[:, 0:2]) / 2.
        gt_wh = gt_boxes[:, 2:4] - gt_boxes[:, 0:2]
        gt_xy = gt_xy[match_indices]
        gt_wh = gt_wh[match_indices]
        gt_polygons = gt_polygons[match_indices]
        gt_rboxes = gt_rboxes[match_indices]

        priors_xy = self.priors_xy[prior_mask] / self.image_size
        priors_wh = self.priors_wh[prior_mask] / self.image_size
        variances_xy = self.priors_variances[prior_mask, 0:2]
        variances_wh = self.priors_variances[prior_mask, 2:4]

        # compute local offsets for
        offsets = np.zeros((num_priors, 4))
        offsets[prior_mask, 0:2] = (gt_xy - priors_xy) / priors_wh
        offsets[prior_mask, 2:4] = np.log(gt_wh / priors_wh)
        offsets[prior_mask, 0:2] /= variances_xy
        offsets[prior_mask, 2:4] /= variances_wh

        # compute local offsets for quadrilaterals
        offsets_quads = np.zeros((num_priors, 8))
        priors_xy_minmax = np.hstack([priors_xy - priors_wh / 2, priors_xy + priors_wh / 2])
        #ref = np.tile(priors_xy, (1,4))
        ref = priors_xy_minmax[:, (0, 1, 2, 1, 2, 3, 0, 3)]  # corner points
        offsets_quads[prior_mask, :] = (gt_polygons - ref) / np.tile(priors_wh, (1, 4)) / np.tile(variances_xy, (1, 4))

        # compute local offsets for rotated bounding boxes
        offsets_rboxs = np.zeros((num_priors, 5))
        offsets_rboxs[prior_mask, 0:2] = (gt_rboxes[:, 0:2] - priors_xy) / priors_wh / variances_xy
        offsets_rboxs[prior_mask, 2:4] = (gt_rboxes[:, 2:4] - priors_xy) / priors_wh / variances_xy
        offsets_rboxs[prior_mask, 4] = np.log(gt_rboxes[:, 4] / priors_wh[:, 1]) / variances_wh[:, 1]

        return np.concatenate([offsets, offsets_quads, offsets_rboxs, confidence], axis=1)

    def decode(self, model_output, confidence_threshold=0.01, keep_top_k=200, fast_nms=True, sparse=True):
        # calculation is done with normalized sizes
        # mbox_loc, mbox_quad, mbox_rbox, mbox_conf
        # 4,8,5,2
        # boxes, quad, rboxes, confs, labels
        # 4,8,5,1,1

        prior_mask = model_output[:, 17:] > confidence_threshold

        if sparse:
            # compute boxes only if the confidence is high enough and the class is not background
            mask = np.any(prior_mask[:, 1:], axis=1)
            prior_mask = prior_mask[mask]
            mask = np.ix_(mask)[0]
            model_output = model_output[mask]
            priors_xy = self.priors_xy[mask] / self.image_size
            priors_wh = self.priors_wh[mask] / self.image_size
            priors_variances = self.priors_variances[mask, :]
        else:
            priors_xy = self.priors_xy / self.image_size
            priors_wh = self.priors_wh / self.image_size
            priors_variances = self.priors_variances

        offsets = model_output[:, :4]
        offsets_quads = model_output[:, 4:12]
        offsets_rboxs = model_output[:, 12:17]
        confidence = model_output[:, 17:]

        priors_xy_minmax = np.hstack([priors_xy - priors_wh / 2, priors_xy + priors_wh / 2])
        ref = priors_xy_minmax[:, (0, 1, 2, 1, 2, 3, 0, 3)]  # corner points
        variances_xy = priors_variances[:, 0:2]
        variances_wh = priors_variances[:, 2:4]

        num_priors = offsets.shape[0]
        num_classes = confidence.shape[1]

        # compute bounding boxes from local offsets
        boxes = np.empty((num_priors, 4))
        offsets = offsets * priors_variances
        boxes_xy = priors_xy + offsets[:, 0:2] * priors_wh
        boxes_wh = priors_wh * np.exp(offsets[:, 2:4])
        boxes[:, 0:2] = boxes_xy - boxes_wh / 2.  # xmin, ymin
        boxes[:, 2:4] = boxes_xy + boxes_wh / 2.  # xmax, ymax
        boxes = np.clip(boxes, 0.0, 1.0)

        # do non maximum suppression
        results = []
        for c in range(1, num_classes):
            mask = prior_mask[:, c]
            boxes_to_process = boxes[mask]
            if len(boxes_to_process) > 0:
                confs_to_process = confidence[mask, c]

                if fast_nms:
                    idx = non_maximum_suppression(
                        boxes_to_process, confs_to_process,
                        self.nms_thresh, self.nms_top_k)
                else:
                    idx = non_maximum_suppression_slow(
                        boxes_to_process, confs_to_process,
                        self.nms_thresh, self.nms_top_k)

                good_boxes = boxes_to_process[idx]
                good_confs = confs_to_process[idx][:, None]
                labels = np.ones((len(idx), 1)) * c

                good_quads = ref[mask][idx] + offsets_quads[mask][idx] * np.tile(priors_wh[mask][idx] * variances_xy[mask][idx], (1, 4))

                good_rboxs = np.empty((len(idx), 5))
                good_rboxs[:, 0:2] = priors_xy[mask][idx] + offsets_rboxs[mask][idx, 0:2] * priors_wh[mask][idx] * variances_xy[mask][idx]
                good_rboxs[:, 2:4] = priors_xy[mask][idx] + offsets_rboxs[mask][idx, 2:4] * priors_wh[mask][idx] * variances_xy[mask][idx]
                good_rboxs[:, 4] = np.exp(offsets_rboxs[mask][idx, 4] * variances_wh[mask][idx, 1]) * priors_wh[mask][idx, 1]

                c_pred = np.concatenate((good_boxes, good_quads, good_rboxs, good_confs, labels), axis=1)
                results.extend(c_pred)
        if len(results) > 0:
            results = np.array(results)
            order = np.argsort(-results[:, 17])
            results = results[order]
            results = results[:keep_top_k]
        else:
            results = np.empty((0, 6))
        self.results = results
        return results

    def plot_results(self, results=None, classes=None, show_labels=False, gt_data=None, confidence_threshold=None):
        if results is None:
            results = self.results
        if confidence_threshold is not None:
            mask = results[:, 17] > confidence_threshold
            results = results[mask]
        if classes is not None:
            colors = plt.cm.hsv(np.linspace(0, 1, len(classes) + 1)).tolist()
        ax = plt.gca()
        im = plt.gci()
        h, w = im.get_size()

        # draw ground truth
        if gt_data is not None:
            for box in gt_data:
                label = np.nonzero(box[4:])[0][0] + 1
                color = 'g' if classes is None else colors[label]
                xy = np.reshape(box[:8], (-1, 2)) * (w, h)
                ax.add_patch(plt.Polygon(xy, fill=True, color=color, linewidth=1, alpha=0.3))

        # draw prediction
        for r in results:
            bbox = r[0:4]
            quad = r[4:12]
            rbox = r[12:17]
            confidence = r[17]
            label = int(r[18])

            plot_box(bbox * (w, h, w, h), box_format='xyxy', color='b')
            plot_box(np.reshape(quad, (-1, 2)) * (w, h), box_format='polygon', color='r')
            plot_box(rbox3_to_polygon(rbox) * (w, h), box_format='polygon', color='g')
            plt.plot(rbox[[0, 2]] * (w, w), rbox[[1, 3]] * (h, h), 'oc', markersize=4)
            if show_labels:
                label_name = label if classes is None else classes[label]
                color = 'r' if classes is None else colors[label]
                xmin, ymin = bbox[:2] * (w, h)
                display_txt = '%0.2f, %s' % (confidence, label_name)
                ax.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})


def multibox_head_separable(source_layers, num_priors, normalizations=None, softmax=True):

    num_classes = 2
    class_activation = 'softmax' if softmax else 'sigmoid'

    mbox_conf = []
    mbox_loc = []
    mbox_quad = []
    mbox_rbox = []
    for i in range(len(source_layers)):
        x = source_layers[i]
        name = x.name.split('/')[0]

        # normalize
        if normalizations is not None and normalizations[i] > 0:
            name = name + '_norm'
            x = Normalize(normalizations[i], name=name)(x)

        # confidence
        name1 = name + '_mbox_conf'
        x1 = SeparableConv2D(num_priors[i] * num_classes, (3, 5), padding='same', name=name1)(x)
        x1 = Flatten(name=name1 + '_flat')(x1)
        mbox_conf.append(x1)

        # location, Delta(x,y,w,h)
        name2 = name + '_mbox_loc'
        x2 = SeparableConv2D(num_priors[i] * 4, (3, 5), padding='same', name=name2)(x)
        x2 = Flatten(name=name2 + '_flat')(x2)
        mbox_loc.append(x2)

        # quadrilateral, Delta(x1,y1,x2,y2,x3,y3,x4,y4)
        name3 = name + '_mbox_quad'
        x3 = SeparableConv2D(num_priors[i] * 8, (3, 5), padding='same', name=name3)(x)
        x3 = Flatten(name=name3 + '_flat')(x3)
        mbox_quad.append(x3)

        # rotated rectangle, Delta(x1,y1,x2,y2,h)
        name4 = name + '_mbox_rbox'
        x4 = SeparableConv2D(num_priors[i] * 5, (3, 5), padding='same', name=name4)(x)
        x4 = Flatten(name=name4 + '_flat')(x4)
        mbox_rbox.append(x4)

    mbox_conf = concatenate(mbox_conf, axis=1, name='mbox_conf')
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation(class_activation, name='mbox_conf_final')(mbox_conf)

    mbox_loc = concatenate(mbox_loc, axis=1, name='mbox_loc')
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

    mbox_quad = concatenate(mbox_quad, axis=1, name='mbox_quad')
    mbox_quad = Reshape((-1, 8), name='mbox_quad_final')(mbox_quad)

    mbox_rbox = concatenate(mbox_rbox, axis=1, name='mbox_rbox')
    mbox_rbox = Reshape((-1, 5), name='mbox_rbox_final')(mbox_rbox)

    predictions = concatenate([mbox_loc, mbox_quad, mbox_rbox, mbox_conf], axis=2, name='predictions')

    return predictions


def TBPP512_dense_separable(input_shape=(512, 512, 3), softmax=True):
    """TextBoxes++512 architecture with dense blocks and separable convolution.
    """

    # custom body
    x = input_tensor = Input(shape=input_shape)
    source_layers = ssd512_dense_separable_body(x)

    num_maps = len(source_layers)

    # Add multibox head for classification and regression
    num_priors = [14] * num_maps
    normalizations = [1] * num_maps
    output_tensor = multibox_head_separable(source_layers, num_priors, normalizations, softmax)
    model = Model(input_tensor, output_tensor)

    # parameters for prior boxes
    model.image_size = input_shape[:2]
    model.source_layers = source_layers

    model.aspect_ratios = [[1, 2, 3, 5, 1 / 2, 1 / 3, 1 / 5] * 2] * num_maps
    #model.shifts = [[(0.0, 0.0)] * 7 + [(0.0, 0.5)] * 7] * num_maps
    model.shifts = [[(0.0, -0.25)] * 7 + [(0.0, 0.25)] * 7] * num_maps
    model.special_ssd_boxes = False
    model.scale = 0.5

    return model
