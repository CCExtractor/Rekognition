from enum import IntEnum


class ImageFrNetworkChoices(IntEnum):
    MTCNN = 1
    RetinaFace = 2

    @classmethod
    def choices(cls):
        return [(key.value, key.name) for key in cls]


def get_class_names(file_name):
    return [c.strip() for c in open(file_name).readlines()]
