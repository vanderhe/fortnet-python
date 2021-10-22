#!/usr/bin/env python3
from distutils.core import setup


setup(
    name='fortnet-python',
    version='0.3',
    description='Python Tools for the Fortnet Software Package',
    author='T. W. van der Heide',
    url='https://github.com/vanderhe/fortnet-python',
    platforms="platform independent",
    package_dir={'': 'src'},
    packages=['fortformat'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    long_description='''
Python Tools for the Fortnet Software Package
---------------------------------------------
fortnet-python provides tools to generate compatible datasets as well as extract
results obtained by the neural network implementation Fortnet.
''',
    requires=['pytest', 'numpy', 'h5py', 'ase']
)
