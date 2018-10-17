from setuptools import setup
import os
import codecs

PACKAGE = 'sru'


def readme():
    """ Return the README text.
    """
    with codecs.open('README.md', encoding='utf-8') as fh:
        return fh.read()


def get_version():
    """ Gets the current version of the package.
    """
    version_py = os.path.join(os.path.dirname(__file__), 'sru/version.py')
    with open(version_py) as fh:
        for line in fh:
            if line.startswith('__version__'):
                return line.split('=')[-1].strip() \
                    .replace('"', '').replace("'", '')
    raise ValueError('Failed to parse version from: {}'.format(version_py))


def get_requirements():
    with open('requirements.txt') as fh:
        lines = fh.readlines()
    lines = [line.strip() for line in lines]
    return [line for line in lines if line]


setup(
    # Package information
    name=PACKAGE,
    version=get_version(),
    description='Simple Recurrent Units for Highly Parallelizable Recurrence',
    long_description=readme(),
    long_description_content_type="text/markdown",  # make pypi render long description as markdown
    keywords='deep learning rnn lstm cudnn sru fast pytorch torch',
    classifiers=[
    ],

    # Author information
    url='https://github.com/taolei87/sru',
    author='Tao Lei, Yu Zhang, Sida I. Wang, Hui Dai and Yoav Artzi',
    author_email='tao@asapp.com',
    license='MIT',

    # What is packaged here.
    packages=['sru'],

    # What to include
    package_data={
        '': ['*.txt', '*.rst', '*.md', '*.cpp']
    },

    # Dependencies
    install_requires=['torch>=0.4.1', 'ninja'],
    extras_require={
        'cuda': ["cupy", "pynvrtc"],
    },
    dependency_links=[
    ],

    zip_safe=False
)
