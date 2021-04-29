from django.test import TestCase
from django.core.files import File
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
settings.configure()
from rest_framework.test import APIClient  # noqa: E402


class TestImageFr(TestCase):

    def setUp(self):

        super(TestImageFr, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/t1.png', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.png", file1.read(), content_type='multipart/form-data')
        file2 = File(open('tests/testdata/t2.jpeg', 'rb'))
        self.uploaded_file2 = SimpleUploadedFile("temp2.jpeg", file2.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/image/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)
        response2 = self.client.post('/api/image/', {'file': self.uploaded_file2})
        self.assertEqual(status.HTTP_200_OK, response2.status_code)

    def test_get(self):

        response1 = self.client.get('/api/image/')
        self.assertEqual(status.HTTP_200_OK, response1.status_code)


class TestNsfwRecognise(TestCase):

    def setUp(self):

        super(TestNsfwRecognise, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/t1.png', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.png", file1.read(), content_type='multipart/form-data')
        file2 = File(open('tests/testdata/t2.jpeg', 'rb'))
        self.uploaded_file2 = SimpleUploadedFile("temp2.jpeg", file2.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/nsfw/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)
        response2 = self.client.post('/api/nsfw/', {'file': self.uploaded_file2})
        self.assertEqual(status.HTTP_200_OK, response2.status_code)


class TestEmbedding(TestCase):

    def setUp(self):

        super(TestEmbedding, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/t1.png', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.png", file1.read(), content_type='multipart/form-data')
        file2 = File(open('tests/testdata/t2.jpeg', 'rb'))
        self.uploaded_file2 = SimpleUploadedFile("temp2.jpeg", file2.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/embed/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)
        response2 = self.client.post('/api/embed/', {'file': self.uploaded_file2})
        self.assertEqual(status.HTTP_200_OK, response2.status_code)

    def test_get(self):

        response1 = self.client.get('/api/embed/')
        self.assertEqual(status.HTTP_200_OK, response1.status_code)


class TestSimilarFace(TestCase):

    def setUp(self):

        super(TestSimilarFace, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/t1.png', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("file.png", file1.read(), content_type='multipart/form-data')
        file2 = File(open('tests/testdata/t2.jpeg', 'rb'))
        self.uploaded_file2 = SimpleUploadedFile("compareImage.jpeg", file2.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/simface/', {'file': self.uploaded_file1, 'compareImage': self.uploaded_file2})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)

    def test_get(self):

        response1 = self.client.get('/api/simface/')
        self.assertEqual(status.HTTP_200_OK, response1.status_code)


class TestObjectDetect(TestCase):

    def setUp(self):

        super(TestObjectDetect, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/t1.png', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.png", file1.read(), content_type='multipart/form-data')
        file2 = File(open('tests/testdata/t2.jpeg', 'rb'))
        self.uploaded_file2 = SimpleUploadedFile("temp2.jpeg", file2.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/objects/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)
        response2 = self.client.post('/api/objects/', {'file': self.uploaded_file2})
        self.assertEqual(status.HTTP_200_OK, response2.status_code)


class TestSceneText(TestCase):

    def setUp(self):

        super(TestSceneText, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/t3.jpeg', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.jpeg", file1.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/scenetext/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)


class TestSceneDetect(TestCase):

    def setUp(self):

        super(TestSceneDetect, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/t3.jpeg', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.jpeg", file1.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/scenedetect/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)
