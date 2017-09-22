###############################################################################
from setuptools import setup, find_packages
import shutil
import os
import filecmp

PACKAGE = 'sru'

################################################################################
def readme():
    """ Return the README text.
    """
    with open('README.md') as fh:
        return fh.read()

################################################################################
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

################################################################################
def get_requirements():
    with open('requirements.txt') as fh:
        lines = fh.readlines()
    lines = [line.strip() for line in lines]
    return [line for line in lines if line]

################################################################################
if not os.path.isfile('sru/cuda_functional.py'):
    shutil.copy('cuda_functional.py', 'sru')
    needs_delete = True
elif not filecmp.cmp(
        'sru/cuda_functional.py', 'cuda_functional.py', shallow=False
    ):
    raise ValueError('Running setup would overwrite the file '
        '"sru/cuda_functional.py". Ensure that any changes to the file are '
        'present in "./cuda_functional.py", delete "sru/cuda_functional.py", '
        'and then try again.')
else:
    needs_delete = False

try:
    setup(
        # Package information
        name=PACKAGE,
        version=get_version(),
        description='Training RNNs as Fast as CNNs',
        long_description=readme(),
        keywords='deep learning rnn lstm cudnn sru fast',
        classifiers=[
        ],

        # Author information
        url='https://github.com/taolei87/sru',
        author='Tao Lei, Yu Zhang',
        author_email='tao@asapp.com',
        license='MIT',

        # What is packaged here.
        packages=['sru'],

        # What to include
        package_data={
            '': ['*.txt', '*.rst', '*.md']
        },

        # Dependencies
        install_requires=get_requirements(),
        dependency_links=[
        ],

        zip_safe=False
    )
finally:
    if needs_delete:
        os.unlink('sru/cuda_functional.py')

#### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
