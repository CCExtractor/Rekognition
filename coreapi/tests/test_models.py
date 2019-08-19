from django.test import TestCase
from coreapi.models import (InputVideo, InputImage)


class InputVideoTest(TestCase):

    def setUp(self):
        pass

    def create_video_object(self):
        return InputVideo.objects.create(title='VideoUnittest', id=1)

    def test_video_object(self):
        expected_video_object = InputVideo.objects.get(id=1)
        expected_object_title = f'{expected_video_object.title}'
        self.assertEqual(expected_object_title, 'VideoUnittest')


class InputImageTest(TestCase):

    def setUp(self):
        pass

    def create_image_object(self):
        return InputImage.objects.create(title='ImageUnittest', id=2)

    def test_image_object(self):
        expected_image_object = InputImage.objects.get(id=2)
        expected_object_title = f'{expected_image_object.title}'
        self.assertEqual(expected_object_title, 'ImageUnittest')
