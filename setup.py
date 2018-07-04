# -*- coding: utf-8 -*-
#
# setup.py
#
# This module is part of skxtend.
#

"""
Setup script for skxtend.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


from setuptools import setup, find_packages


PACKAGE_NAME = 'skxtend'
VERSION = '0.1.0'
KEYWORDS = 'machine learning, data analysis'
TESTS_REQUIRE = ['pytest', 'mock']


def readme():
    """Return the contents of the README.md file."""

    with open('README.md') as freadme:
        return freadme.read()


def requirements():
    """Return the contents of the REQUIREMENTS.txt file."""

    with open('REQUIREMENTS.txt', 'r') as freq:
        return freq.read().splitlines()


def license():
    """Return the contents of the LICENSE.txt file."""

    with open('LICENSE.txt') as flicense:
        return flicense.read()


setup(
    author="Severin Langberg",
    author_email="Langberg91@gmail.com",
    description="Tools for preforming data analysis and predictive modelling.",
    url='https://github.com/GSEL9/ds-tools',
    install_requires=requirements(),
    long_description=readme(),
    license=license(),
    name=PACKAGE_NAME,
    version=VERSION,
    packages=find_packages(exclude=['test']),
    setup_requires=['pytest-runner'],
    tests_require=TESTS_REQUIRE,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Environment :: Console',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
