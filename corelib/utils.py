from enum import IntEnum
from logger.logging import RekogntionLogger


logger = RekogntionLogger(name="utils")


def bb_to_cv(box):
    """     Bounding Box to OpenCV style coordinates
    Args:
            *   box: coordinates of the bounding box
    Workflow:
            *   Bounding Box style coordinates are taken as input
                 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            *   List items are unpacked to obtain coordinates
            *   Minimum and Maximum coordinates are calculated to
                obtain the top left and bottom right coordinates.
    Returns:
            *   OpenCV style coordinates
    """

    logger.info(msg="bb_to_cv called")
    x1, y1 = box[0][0], box[0][1]
    x2, y2 = box[1][0], box[1][1]
    x3, y3 = box[2][0], box[2][1]
    x4, y4 = box[3][0], box[3][1]
    top_left_x = min([x1, x2, x3, x4])
    top_left_y = min([y1, y2, y3, y4])
    bot_right_x = max([x1, x2, x3, x4])
    bot_right_y = max([y1, y2, y3, y4])
    return top_left_x, top_left_y, bot_right_x, bot_right_y


class ImageFrNetworkChoices(IntEnum):
    MTCNN = 1
    RetinaFace = 2

    @classmethod
    def choices(cls):
        return [(key.value, key.name) for key in cls]


def get_class_names(file_name):
    """     Read contents of file and return list of lines
    Args:
            *   file_name: path of file to be read
    Workflow:
            *   Reads file line by line and removes
                the leading and trailing characters
                in each line
    Returns:
            *   list of lines in the file
    """

    logger.info(msg="get_class_names called")
    return [c.strip() for c in open(file_name).readlines()]


def get_classes(file_name):
    """     Read contents of file and return tuple of words
    Args:
            *   file_name: path of file to be read
    Workflow:
            *   Reads file line by line and removes
                the leading and trailing characters
                in each line
            *   Splits each line into a list and returns
                letters occuring after the 3rd elemnt in
                the of the first element of the list
            *   converts list to tuple
    Returns:
            *   tuple of required words from the file
    """

    logger.info(msg="get_classes called")
    classes = []
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    return tuple(classes)
