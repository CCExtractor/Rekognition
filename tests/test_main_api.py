from django.test import TestCase
from corelib.main_api import object_detect
import requests
import json

class TestObjectDetect(TestCase):
	"""docstring for TestObjectDetect"""
	def setUp(self):
		super(TestObjectDetect,self).setUp()
		self.url = "http://127.0.0.1:8000/api/objects/"
		self.obj1 = open("./tests/testdata/bicycle.jpg",'rb')
		self.label1 = "bicycle"

	def test_single_object_detection(self):
		test_obj1 = {'file': self.obj1 }
		result = json.loads(requests.post(self.url, files=test_obj1).text)
		# result = json.loads(response.text)
		self.assertEqual(result["Objects"][0][2]["Label"],self.label1)

		

