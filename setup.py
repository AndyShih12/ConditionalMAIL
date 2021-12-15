#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='conditionalMAIL',
      version='0.0.1',
      description='conditionalMAIL',
      author='Andy Shih',
      author_email='andyshih@cs.stanford.edu',
      packages=find_packages(),
      install_requires=[
        'flask',
        'numpy',
        'torch',
        'tqdm',
        'garage@git+https://github.com/rlworkgroup/garage@v2020.06.0',
        'hanabi-learning-environment@git+https://github.com/deepmind/hanabi-learning-environment#4210e0edadb636fdccf528f75943df685085878b',
      ],
    )
