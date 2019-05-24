# from django.test import TestCase  # NOQA

# Create your tests here.

import unittest


class SimpleTest(unittest.TestCase):

    # Returns True or False.

    def test(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
