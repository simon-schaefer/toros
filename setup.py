from setuptools import find_packages
from setuptools import setup


setup(
    name='toros',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['torch'],  # see requirements.txt
    author='Simon Schaefer',
    author_email='simon.k.schaefer@gmail.com',
    license='Apache',
    url='https://github.com/simon-schaefer/toros',
    description='Torch <-> ROS Interface',
)
