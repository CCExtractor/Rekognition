#! /usr/bin/env python3.6
"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import os,shutil
import subprocess
import sys
import setuptools
from setuptools.dist import Distribution
from distutils.sysconfig import get_python_lib


# This is a hack around python wheels not including the adaptor.so library.
class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
with open('../requirements.txt') as f:
    required = f.read().splitlines()
if(sys.version[2]=='6'):
	if subprocess.call(['make', '--always-make','-C', BASE_DIR]) != 0:
	    raise RuntimeError('Cannot compile lanms in the directory: {}'.format(BASE_DIR))

	setuptools.setup(
	    name='lanms',

	    version='1.0.2',

	    description='Locality-Aware Non-Maximum Suppression',

	    # The project's main homepage.
	    url='https://github.com/Parquery/lanms',

	    # Author details
	    author='argmen (boostczc@gmail.com) is code author, '
		   'Dominik Walder (dominik.walder@parquery.com) and Marko Ristin (marko@parquery.com) only packaged the code',
	    author_email='devs@parquery.com',

	    # Choose your license
	    license='GNU General Public License v3.0',

	    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
	    classifiers=[
		'Development Status :: 5 - Production/Stable',

		'Intended Audience :: Developers',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',

		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

		'Programming Language :: Python :: 3.5',
	    ],

	    keywords='locality aware non-maximum suppression',

	    packages=setuptools.find_packages(exclude=[]),

	    install_requires=required,

	    include_package_data=True,
	    distclass=BinaryDistribution,
	)
	def copytree(src, dst, symlinks=False, ignore=None):
		for item in os.listdir(src):
			s = os.path.join(src, item)
			d = os.path.join(dst, item)
			if os.path.isdir(s):
				shutil.copytree(s, d, symlinks, ignore)
			else:
				shutil.copy2(s, d)
	os.mkdir(get_python_lib()+"/lanms")
	copytree("lanms",get_python_lib()+"/lanms")
else:
	subprocess.check_call([sys.executable, "-m", "pip", "install", "lanms"])
