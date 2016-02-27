#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('pyseqsample')
    return config

def main():
    from numpy.distutils.core import setup
    setup(name='pyseqsample',
          version='0.1',
          description='Python functions for studying sequential sampling models',
          author='Gilles de Hollander',
          author_email='g.dehollander@uva.nl',
          url='http://www.gillesdehollander.nl',
          packages=['pyseqsample'],
          configuration=configuration
         )

if __name__ == '__main__':
    main()
