#!/usr/bin/env python

import sys
py_version = f"{sys.version_info.major}{sys.version_info.minor}"

from setuptools import find_packages
from setuptools import setup

install_requires = ["yacs",
                    "s3fs",
                    "mpi4py",
                    "opencv-python",
                    "pycocotools @ git+https://github.com/johnbensnyder/cocoapi.git@nvidia/master#subdirectory=PythonAPI"]

#pip install 'git+https://github.com/NVIDIA/cocoapi.git@nvidia/master#subdirectory=PythonAPI'

setup(
    name="sagemakercv",
    version="0.1",
    author="jbsnyder",
    url="https://github.com/aws-samples/amazon-sagemaker-cv",
    description="Computer vision in TensorFlow with Amazon Sagemaker",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=install_requires
)
