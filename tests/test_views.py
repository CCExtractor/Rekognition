from django.test import TestCase
from rest_framework.test import APIClient
from django.core.files import File
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status


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
        response2 = self.client.get('/api/image/')
        self.assertEqual(status.HTTP_200_OK, response2.status_code)
