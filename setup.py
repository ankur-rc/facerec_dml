"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()



setup(
    name='face_trigger',  # Required
    version='0.2.1',  # Required
    description='Face-recognition framework',  # Required
    long_description=long_description,  # Optional
    url='https://github.com/SofturaInternal/face-trigger',  # Optional
    author='Ankur Roy Chowdhury (Softura)',  # Optional
    author_email='ankurrc@softura.com',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Computer Vision :: Face Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7'
    ],
    keywords='computer-vision face-recognition',  # Optional
    packages=find_packages(
        exclude=['contrib', 'docs', 'tests', 'notebooks', 'examples']),  # Required
    install_requires=["dlib == 19.13.1",
                      "numpy == 1.14.5",
                      "opencv_python == 3.4.1.15",
                      "tqdm == 4.23.4",
                      "scikit_learn == 0.19.1",
                      "scipy == 1.1.0"],  # Optional
    package_data={  # Optional
        'face_trigger': ['pre_trained/*.dat'],
    }
)
