from setuptools import setup, find_packages

# Compile some code for performance
from Cython.Build import cythonize

# Boilerplate for integrating with PyTest
from setuptools.command.test import test
import sys
import os


class PyTest(test):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

# The actual setup metadata
setup(
    name='antivirals',
    version='0.0.2',
    description='Finding antivirals for the novel coronavirus.',
    long_description=open("README.rst").read(),
    keywords='machine_learning artificial_intelligence medicine devops',
    author='JJ Ben-Joseph',
    author_email='jbenjoseph@iqt.org',
    python_requires='>=3.7',
    url='https://www.github.com/bnext-iqt/antivirals',
    license='Apache',
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent'
    ],
    packages=find_packages(),
    install_requires=['tqdm', 'sqlalchemy', 'numpy', 'scikit-learn', 'pandas',
                      'fire', 'gensim'],
    extras_require={
        'optim': ['sigopt']
    },
    tests_require=['pytest'],
    ext_modules = cythonize('antivirals/parser.py', language_level='3'),
    entry_points={
        'console_scripts': [
            'antivirals = antivirals.__main__:main',
        ],
    },
    cmdclass={'test': PyTest}
)
