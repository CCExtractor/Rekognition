from django.test import TestCase

from coreapi.models import InputImage, InputVideo, SimilarFaceInImage, InputEmbed, NameSuggested

import uuid


class TestInputImage(TestCase):

    def setUp(self):
        super(TestInputImage, self).setUp()
        self.obj1 = InputImage.objects.create(title='test_name_1')
        self.obj2 = InputImage.objects.create(title='test_name_2', is_processed=True)

    def test_inputimage_model_creation(self):

        test_name_1 = InputImage.objects.get(title='test_name_1')
        self.assertEqual(test_name_1.title, self.obj1.title)
        self.assertEqual(test_name_1.is_processed, False)
        self.assertEqual(test_name_1.created_on, self.obj1.created_on)

        test_name_2 = InputImage.objects.get(title='test_name_2')
        self.assertEqual(test_name_2.title, self.obj2.title)
        self.assertEqual(test_name_2.is_processed, True)
        self.assertEqual(test_name_2.created_on, self.obj2.created_on)


class TestInputVideo(TestCase):

    def setUp(self):
        super(TestInputVideo, self).setUp()
        self.obj1 = InputVideo.objects.create(title='test_name_1')
        self.obj2 = InputVideo.objects.create(title='test_name_2', is_processed=True)

    def test_inputvideo_model_creation(self):

        test_name_1 = InputVideo.objects.get(title='test_name_1')
        self.assertEqual(test_name_1.title, self.obj1.title)
        self.assertEqual(test_name_1.is_processed, False)
        self.assertEqual(test_name_1.created_on, self.obj1.created_on)

        test_name_2 = InputVideo.objects.get(title='test_name_2')
        self.assertEqual(test_name_2.title, self.obj2.title)
        self.assertEqual(test_name_2.is_processed, True)
        self.assertEqual(test_name_2.created_on, self.obj2.created_on)


class TestInputEmbed(TestCase):

    def setUp(self):
        super(TestInputEmbed, self).setUp()
        self.id1 = uuid.uuid4().hex
        self.id2 = uuid.uuid4().hex

        self.obj1 = InputEmbed.objects.create(title='test_name_1', id=self.id1, fileurl='test_path_1')
        self.obj2 = InputEmbed.objects.create(title='test_name_2', id=self.id2, fileurl='test_path_2')

    def test_model_creation(self):

        test_name_1 = InputEmbed.objects.get(title='test_name_1')
        self.assertEqual(test_name_1.title, self.obj1.title)
        self.assertEqual(test_name_1.fileurl, self.obj1.fileurl)
        self.assertEqual(test_name_1.created_on, self.obj1.created_on)

        test_name_2 = InputEmbed.objects.get(title='test_name_2')
        self.assertEqual(test_name_2.title, self.obj2.title)
        self.assertEqual(test_name_2.fileurl, self.obj2.fileurl)
        self.assertEqual(test_name_2.created_on, self.obj2.created_on)


class TestNameSuggested(TestCase):

    def setUp(self):
        super(TestNameSuggested, self).setUp()

        self.embed = InputEmbed.objects.create(title='test_name_1', id=uuid.uuid4().hex, fileurl='test_path_1')
        self.obj1 = NameSuggested.objects.create(suggested_name=self.embed.title, feedback=self.embed)

    def test_model_creation(self):

        test_name_1 = NameSuggested.objects.get(suggested_name='test_name_1')
        self.assertEqual(test_name_1.suggested_name, self.embed.title)
        self.assertEqual(test_name_1.upvote, 0)
        self.assertEqual(test_name_1.downvote, 0)
        self.assertEqual(test_name_1.feedback, self.embed)


class TestSimilarFaceInImage(TestCase):

    def setUp(self):
        super(TestSimilarFaceInImage, self).setUp()
        self.obj1 = SimilarFaceInImage.objects.create(title='test_title')

    def test_model_creation(self):

        test_name_1 = SimilarFaceInImage.objects.get(title='test_title')
        self.assertEqual(test_name_1.title, self.obj1.title)
        self.assertEqual(test_name_1.similarwith, self.obj1.similarwith)
        self.assertEqual(test_name_1.created_on, self.obj1.created_on)
