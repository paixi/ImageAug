#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ImageAug',      
    version='0.1.0post',
    author='Paixi',
    author_email='paixi@protonmail.com',
    description='Image augmentation for PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/paixi/ImageAug',
    packages=['imageaug'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ]
)
