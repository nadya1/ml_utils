__date__ = "04/01/2017"

# !/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='ml_utils',
    version='1.0',
    description='Machine Learning utils from graphlab & numpy',
    autor='Nadya Ortiz',
    packages=find_packages(where='src'),
    install_requires=[
        'ipython',
        'matplotlib',
        'numpy',
        'pandas',
    ],
    include_package_data=True,
)
