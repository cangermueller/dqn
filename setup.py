from glob import glob
import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='dqn',
      version='0.0.1',
      description='Deep Q-Networks',
      long_description=read('README.md'),
      author='Christof Angermueller',
      author_email='cangermueller@gmail.com',
      license="MIT",
      packages=find_packages(),
      scripts=glob('./scripts/*.py')
      )
