from enum import IntEnum
from logger.logging import RekogntionLogger


logger = RekogntionLogger(name="main_api")


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
