from __future__ import absolute_import

from glob import glob
import os
from os.path import basename
from os.path import splitext

from setuptools import find_packages, setup

setup(
    name='custom_lightgbm_framework',
    version='1.0.0',
    description='Custom framework container training package for LightGBM.',
    keywords="custom framework container training package SageMaker LightGBM",

    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    
    install_requires=['sagemaker-training==3.4.1']
)
