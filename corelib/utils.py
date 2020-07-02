from enum import IntEnum


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

    x1, y1 = box[0][0], box[0][1]
    x2, y2 = box[1][0], box[1][1]
    x3, y3 = box[2][0], box[2][1]
    x4, y4 = box[3][0], box[3][1]
    top_left_x = min([x1,x2,x3,x4])
    top_left_y = min([y1,y2,y3,y4])
    bot_right_x = max([x1,x2,x3,x4])
    bot_right_y = max([y1,y2,y3,y4])
    return top_left_x, top_left_y, bot_right_x, bot_right_y


class ImageFrNetworkChoices(IntEnum):
    MTCNN = 1
    RetinaFace = 2

    @classmethod
    def choices(cls):
        return [(key.value, key.name) for key in cls]
