import os
from setuptools import setup, find_packages
from typing import AnyStr


def read(fname: str) -> AnyStr:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="svlearn-bootcamp",
    version="1.0.0",
    author="Asif Qamar",
    author_email="asif@supportvectors.com",
    description="Text-extraction from given file.",
    url="https://packages.python.org/svlearn-bootcamp",
    packages=find_packages(),
    long_description=read('README'),
    classifiers=["Operating System::OS Independent"],
)
