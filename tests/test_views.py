from django.test import TestCase
from django.core.files import File
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
from rest_framework.test import APIClient  # noqa: E402


class TestImageFr(TestCase):

    def setUp(self):
        print("Testing ImageFr")
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


class TestVideoFr(TestCase):

    def setUp(self):
        print("Testing TestVideoFr")

        super(TestVideoFr, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/test1.mp4', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.mp4", file1.read(), content_type='multipart/form-data')
        file2 = File(open('tests/testdata/test2.mp4', 'rb'))
        self.uploaded_file2 = SimpleUploadedFile("temp2.mp4", file2.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/video/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)
        response2 = self.client.post('/api/video/', {'file': self.uploaded_file2})
        self.assertEqual(status.HTTP_200_OK, response2.status_code)


class TestAsyncVideoFr(TestCase):

    def setUp(self):
        
        print("Testing TestAsyncVideoFr")
        super(TestAsyncVideoFr, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/test1.mp4', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.mp4", file1.read(), content_type='multipart/form-data')
        file2 = File(open('tests/testdata/test2.mp4', 'rb'))
        self.uploaded_file2 = SimpleUploadedFile("temp2.mp4", file2.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/video/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)
        response2 = self.client.post('/api/video/', {'file': self.uploaded_file2})
        self.assertEqual(status.HTTP_200_OK, response2.status_code)


class TestNsfwRecognise(TestCase):

    def setUp(self):
        print("Testing TestNsfwRecognise")
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


class TestNsfwVideo(TestCase):

    def setUp(self):
        print("Testing TestNsfwVideo")
        super(TestNsfwVideo, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/test3.mp4', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.mp4", file1.read(), content_type='multipart/form-data')
        file2 = File(open('tests/testdata/test4.mp4', 'rb'))
        self.uploaded_file2 = SimpleUploadedFile("temp2.mp4", file2.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/nsfwvideo/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)
        response2 = self.client.post('/api/nsfwvideo/', {'file': self.uploaded_file2})
        self.assertEqual(status.HTTP_200_OK, response2.status_code)


class TestEmbedding(TestCase):

    def setUp(self):
        print("Testing TestEmbedding")
        super(TestEmbedding, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/compareImage.jpeg', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.jpeg", file1.read(), content_type='multipart/form-data')
        file2 = File(open('tests/testdata/compareImage.jpeg', 'rb'))
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
        print("Testing TestSimilarFace")
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


# class TestObjectDetect(TestCase):

#     def setUp(self):

#         super(TestObjectDetect, self).setUp()
#         self.client = APIClient()
#         file1 = File(open('tests/testdata/t1.png', 'rb'))
#         self.uploaded_file1 = SimpleUploadedFile("temp1.png", file1.read(), content_type='multipart/form-data')
#         file2 = File(open('tests/testdata/t2.jpeg', 'rb'))
#         self.uploaded_file2 = SimpleUploadedFile("temp2.jpeg", file2.read(), content_type='multipart/form-data')

#     def test_post(self):

#         response1 = self.client.post('/api/objects/', {'file': self.uploaded_file1})
#         self.assertEqual(status.HTTP_200_OK, response1.status_code)
#         response2 = self.client.post('/api/objects/', {'file': self.uploaded_file2})
#         self.assertEqual(status.HTTP_200_OK, response2.status_code)


# class TestObjectDetectVideo(TestCase):

#     def setUp(self):

#         super(TestObjectDetectVideo, self).setUp()
#         self.client = APIClient()
#         file1 = File(open('tests/testdata/obj1.mp4', 'rb'))
#         self.uploaded_file1 = SimpleUploadedFile("temp1.mp4", file1.read(), content_type='multipart/form-data')
#         file2 = File(open('tests/testdata/obj2.mp4', 'rb'))
#         self.uploaded_file2 = SimpleUploadedFile("temp2.mp4", file2.read(), content_type='multipart/form-data')

#     def test_post(self):

#         response1 = self.client.post('/api/objectsvideo/', {'file': self.uploaded_file1})
#         self.assertEqual(status.HTTP_200_OK, response1.status_code)
#         response2 = self.client.post('/api/objectsvideo/', {'file': self.uploaded_file2})
#         self.assertEqual(status.HTTP_200_OK, response2.status_code)


class TestSceneText(TestCase):

    def setUp(self):
        print("Testing TestSceneText")
        super(TestSceneText, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/t3.jpeg', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.jpeg", file1.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/scenetext/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)


class TestSceneTextVideo(TestCase):

    def setUp(self):
        print("Testing TestSceneTextVideo")
        super(TestSceneTextVideo, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/test3.mp4', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.mp4", file1.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/scenetextvideo/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)


class TestSceneDetect(TestCase):

    def setUp(self):
        print("Testing TestSceneDetect")
        super(TestSceneDetect, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/t3.jpeg', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.jpeg", file1.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/scenedetect/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)


class TestSceneVideo(TestCase):

    def setUp(self):
        print("Testing TestSceneVideo")
        super(TestSceneVideo, self).setUp()
        self.client = APIClient()
        file1 = File(open('tests/testdata/test1.mp4', 'rb'))
        self.uploaded_file1 = SimpleUploadedFile("temp1.mp4", file1.read(), content_type='multipart/form-data')

    def test_post(self):

        response1 = self.client.post('/api/scenevideo/', {'file': self.uploaded_file1})
        self.assertEqual(status.HTTP_200_OK, response1.status_code)
