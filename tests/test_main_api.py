from django.test import TestCase
from corelib.constant import object_detect_api , nsfw_classifier_api , sim_face_search_api
import requests
import json

class TestObjectDetect(TestCase):
	"""docstring for TestObjectDetect"""
	def setUp(self):
		super(TestObjectDetect,self).setUp()
		self.test_obj = {'file': open("./tests/testdata/bicycle.jpg",'rb') }
		self.label = "bicycle"

	def test_single_object_detection(self):
		result = json.loads(requests.post(object_detect_api, files=self.test_obj).text)
		self.assertEqual(result["Objects"][0][2]["Label"],self.label)

class TestNsfwClassifier(TestCase):
	def setUp(self):
		super(TestNsfwClassifier,self).setUp()
		self.test_obj1 = {"file":open("./tests/testdata/nsfw.jpg",'rb')}
		self.test_obj2 = {"file":open("./tests/testdata/drawing.jpg",'rb')}
		self.label1 = "porn"
		self.label2 = "neutral"

	def test_nsfw_classification(self):
		result1 = json.loads(requests.post(nsfw_classifier_api, files=self.test_obj1).text)
		self.assertEqual(result1.get("classes").lower(),self.label1)

		result2 = json.loads(requests.post(nsfw_classifier_api, files=self.test_obj2).text)
		self.assertEqual(result2.get("classes").lower(),self.label2)

class TestSimilarFacceSearch(TestCase):
	def setUp(self):
		super(TestSimilarFacceSearch,self).setUp()
		self.test_obj = {"file":open("./tests/testdata/reference.jpg","rb") , "compareImage":open("./tests/testdata/compare.jpg","rb")}
		self.output = 5

	def test_similar_face_search(self):
		result = json.loads(requests.post(sim_face_search_api,files=self.test_obj).text)
		self.assertEqual(result["result"][1],self.output)
